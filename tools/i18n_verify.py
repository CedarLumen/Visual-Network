# -*- coding: utf-8 -*-
"""独立校验各组的 TSV：行数、中文列逐字节、英文列不含中文，并抽样打印。"""
import io
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
CJK = re.compile('[\u4e00-\u9fff]')
BASE = os.path.join(os.environ['TEMP'], 'i18n')


def read_list(path):
    groups = []
    cur = None
    for line in io.open(path, encoding='utf-8'):
        line = line.rstrip('\n').rstrip('\r')
        if line.startswith('# '):
            cur = (line[2:].strip(), [])
            groups.append(cur)
        elif line:
            cur[1].append(line)
    return groups


bad = 0
for key in sys.argv[1:]:
    lst = os.path.join(BASE, 'list_%s.txt' % key)
    tsv = os.path.join(BASE, 'tr_%s.tsv' % key)
    groups = read_list(lst)
    zh = [s for _f, items in groups for s in items]
    rows = []
    for i, line in enumerate(io.open(tsv, encoding='utf-8')):
        line = line.rstrip('\n').rstrip('\r')
        if not line:
            continue
        parts = line.split('\t')
        if len(parts) != 2:
            print('  [问题] %s:%d 不是两列' % (tsv, i + 1))
            bad += 1
            continue
        rows.append(parts)
    print('组 %s：清单 %d 条，TSV %d 条' % (key, len(zh), len(rows)))
    if len(zh) != len(rows):
        print('  [问题] 条数不一致')
        bad += 1
        continue
    shift = 0
    for i, (a, b) in enumerate(rows):
        if a != zh[i]:
            shift += 1
            if shift <= 3:
                print('  [问题] 第 %d 行中文对不上：清单=%r TSV=%r' % (i + 1, zh[i], a))
        if CJK.search(b):
            print('  [问题] 第 %d 行英文里有中文：%r -> %r' % (i + 1, a, b))
            bad += 1
        if '\t' in b or not b.strip():
            print('  [问题] 第 %d 行英文为空或含 TAB' % (i + 1))
            bad += 1
    if shift:
        print('  [问题] 共 %d 行中文对不上' % shift)
        bad += shift
    print('  抽样：')
    for i in (0, len(rows) // 3, len(rows) // 2, len(rows) - 1):
        print('    %-28s -> %s' % (rows[i][0][:26], rows[i][1][:70]))
print('== 校验结束：%d 处问题 ==' % bad)
