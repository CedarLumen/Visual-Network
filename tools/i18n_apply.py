# -*- coding: utf-8 -*-
"""
界面多语言的落地工具（中文原文 → core::tr("中文", "English")）。

工作流
------
1) 清点：把源码里带中文的字符串字面量按文件列成清单（每行一条，`# 路径` 开头的是文件标记）。
2) 翻译：为每个清单写一份同序的英文清单（一行一条）。
   允许项：`src/core/Names.cpp` 里的中文是「默认模块名/组名本身」（存在图里的数据，不是待翻译的
   界面文案），界面靠 `core::displayName()` 在显示时按语言映射，所以该文件不需要 tr() 包装。
3) make：`python tools/i18n_apply.py make <清单> <英文清单> <输出.tsv>` 合成 TSV（中文<TAB>英文）。
4) apply：`python tools/i18n_apply.py apply <清单> <tsv>` 按清单里的文件归属就地包装源码。
5) check：`python tools/i18n_apply.py check <清单> [清单...]` 检查清单里的中文串是否都已包装进 tr()。

约定
----
- 中文那一份必须与清单逐字节相同（apply 用它精确匹配源码里的字面量）；
- 只包装字符串字面量，注释里的中文不管；
- 已经被 tr(...) 包过的字面量会跳过，所以 apply 可以重复跑；
- 默认语言仍是中文，源码里的中文原文不会被改动。
"""
import io
import os
import re
import sys

CJK = re.compile('[\u4e00-\u9fff]')


def is_tr_open(text, start):
    """start 位置的字面量是不是已经被 tr(...) 包住了（'tr(' 前面不能是标识符字符，避免 substr( 误判）。"""
    i = start - 1
    while i >= 0 and text[i] in ' \t\r\n':
        i -= 1
    if i < 2:
        return False
    if text[i - 2:i + 1] != 'tr(':
        return False
    if i - 3 >= 0:
        prev = text[i - 3]
        if prev.isalnum() or prev == '_' or prev == '.':
            return False
    return True


def scan_spans(text):
    """返回源码里所有字符串字面量的 (起, 止) 位置（跳过注释与字符常量）。"""
    spans = []
    i = 0
    n = len(text)
    while i < n:
        if text.startswith('/*', i):
            j = text.find('*/', i + 2)
            i = n if j < 0 else j + 2
            continue
        if text.startswith('//', i):
            j = text.find('\n', i)
            i = n if j < 0 else j
            continue
        if text[i] == '"':
            j = i + 1
            while j < n:
                if text[j] == '\\':
                    j += 2
                    continue
                if text[j] == '"':
                    break
                j += 1
            spans.append((i, j + 1))
            i = j + 1
            continue
        i += 1
    return spans


def read_list(path):
    """读清单：返回 [(文件, [中文串, ...]), ...]，保持顺序。"""
    groups = []
    cur = None
    for line in io.open(path, encoding='utf-8'):
        line = line.rstrip('\n').rstrip('\r')
        if line.startswith('# '):
            cur = (line[2:].strip(), [])
            groups.append(cur)
        elif line:
            if cur is None:
                raise SystemExit('清单缺少文件标记：%s' % path)
            cur[1].append(line)
    return groups


def read_tsv(path):
    rows = []
    for line in io.open(path, encoding='utf-8'):
        line = line.rstrip('\n').rstrip('\r')
        if not line:
            continue
        parts = line.split('\t')
        if len(parts) != 2:
            raise SystemExit('%s 这一行不是两列：%r' % (path, line))
        rows.append((parts[0], parts[1]))
    return rows


def cmd_make(args):
    lst, en_path, out = args[0], args[1], args[2]
    groups = read_list(lst)
    zh = [s for _f, items in groups for s in items]
    en = [l.rstrip('\n').rstrip('\r') for l in io.open(en_path, encoding='utf-8') if l.strip()]
    if len(zh) != len(en):
        raise SystemExit('条数不一致：清单 %d 条，英文 %d 条' % (len(zh), len(en)))
    with io.open(out, 'w', encoding='utf-8', newline='\n') as f:
        for a, b in zip(zh, en):
            if CJK.search(b):
                raise SystemExit('英文列里有中文：%r' % b)
            f.write('%s\t%s\n' % (a, b))
    print('写出 %s（%d 条）' % (out, len(zh)))


def escape_en(s):
    return s.replace('\\', '\\\\').replace('"', '\\"')


def cmd_apply(args):
    lst, tsv = args[0], args[1]
    groups = read_list(lst)
    rows = read_tsv(tsv)
    zh = [s for _f, items in groups for s in items]
    if len(zh) != len(rows):
        raise SystemExit('条数不一致：清单 %d 条，TSV %d 条' % (len(zh), len(rows)))
    if [a for a, _b in rows] != zh:
        for k, (a, b) in enumerate(zip(zh, [x for x, _y in rows])):
            if a != b:
                raise SystemExit('第 %d 条中文对不上：\n  清单：%r\n  TSV ：%r' % (k + 1, a, b))
    # 逐文件建立 中文 -> 英文 映射
    idx = 0
    total = 0
    for rel, items in groups:
        mapping = {}
        for s in items:
            mapping[s] = rows[idx][1]
            idx += 1
        text = io.open(rel, encoding='utf-8', newline='').read()
        spans = scan_spans(text)
        # C++ 里相邻的字面量会拼接成一句（"前半" "后半"），要整段包成一个 tr()，
        # 否则会变成 tr(...) tr(...) 两个函数调用并列，语法不成立。
        runs = []
        for span in spans:
            if runs and not text[runs[-1][-1][1]:span[0]].strip():
                runs[-1].append(span)
            else:
                runs.append([span])
        hits = 0
        for run in reversed(runs):
            bodies = [text[s + 1:e - 1] for s, e in run]
            known = [b in mapping for b in bodies]
            if not any(known):
                continue
            if all(known):
                en = ''
                for b in bodies:
                    piece = mapping[b]
                    if en and not (en.endswith(' ') or piece.startswith(' ')):
                        en += ' '
                    en += piece
                start, end = run[0][0], run[-1][1]
                zh = text[start:end]  # 原样保留，里面的换行与拼接都不动
                text = text[:start] + 'tr(%s, "%s")' % (zh, escape_en(en)) + text[end:]
                hits += len(run)
            else:
                # 只包住其中一部分：先按单条处理，编译不过的话要手工合并
                for start, end in reversed(run):
                    body = text[start + 1:end - 1]
                    if body not in mapping or is_tr_open(text, start):
                        continue
                    text = (text[:start] + 'tr("%s", "%s")' % (body, escape_en(mapping[body])) +
                            text[end:])
                    hits += 1
        if hits:
            io.open(rel, 'w', encoding='utf-8', newline='').write(text)
        total += hits
        print('%-34s 包装 %3d 处' % (rel, hits))
    print('合计包装 %d 处' % total)


def cmd_check(args):
    bad = 0
    for lst in args:
        for rel, items in read_list(lst):
            text = io.open(rel, encoding='utf-8', newline='').read()
            spans = scan_spans(text)
            for start, end in spans:
                body = text[start + 1:end - 1]
                if not CJK.search(body):
                    continue
                before = text[max(0, start - 8):start].rstrip()
                line_start = text.rfind('\n', 0, start) + 1
                if is_tr_open(text, start):
                    continue  # tr("中文", ...) 的第一个参数
                if before.endswith(',') and 'tr(' in text[line_start:start]:
                    continue  # tr("中文", "English") 的第二个参数
                bad += 1
                line = text.count('\n', 0, start) + 1
                print('%s:%d 未包装的中文串：%s' % (rel, line, body[:60]))
    if bad:
        print('== 还有 %d 处未包装 ==' % bad)
        return 1
    print('== 清单里的中文串都已包装进 tr() ==')
    return 0


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    cmd = sys.argv[1]
    if cmd == 'make':
        cmd_make(sys.argv[2:])
    elif cmd == 'apply':
        cmd_apply(sys.argv[2:])
    elif cmd == 'check':
        sys.exit(cmd_check(sys.argv[2:]))
    else:
        raise SystemExit(__doc__)


main()
