# -*- coding: utf-8 -*-
"""把 MNIST 测试集解成 C++ 自检能直接读的二进制：
    tools/out/mnist_test.bin
        uint32 magic   = 0x4D4E4953 ('MNIS')
        uint32 count   = 10000
        uint32 rows    = 28
        uint32 cols    = 28
        uint8  labels[count]
        uint8  pixels[count * rows * cols]   每张图按行优先

只用标准库（gzip + struct），不依赖 numpy。数据来自 tools/dataset/ 里随工程发布的两个
MNIST 压缩文件。跑一次生成即可，nne_tests 会读它做 10000 张全量评估。
"""

import gzip
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(HERE, 'dataset')
OUT = os.path.join(HERE, 'out')

MAGIC = 0x4D4E4953


def read_idx_images(path):
    with gzip.open(path, 'rb') as f:
        data = f.read()
    magic, count, rows, cols = struct.unpack('>IIII', data[:16])
    if magic != 2051:
        raise RuntimeError('%s 不是 MNIST 图像文件（magic=%d）' % (path, magic))
    need = 16 + count * rows * cols
    if len(data) < need:
        raise RuntimeError('%s 长度不足：%d < %d' % (path, len(data), need))
    return count, rows, cols, data[16:need]


def read_idx_labels(path):
    with gzip.open(path, 'rb') as f:
        data = f.read()
    magic, count = struct.unpack('>II', data[:8])
    if magic != 2049:
        raise RuntimeError('%s 不是 MNIST 标签文件（magic=%d）' % (path, magic))
    return count, data[8:8 + count]


def main():
    img = os.path.join(DATASET, 't10k-images-idx3-ubyte.gz')
    lab = os.path.join(DATASET, 't10k-labels-idx1-ubyte.gz')
    for p in (img, lab):
        if not os.path.exists(p):
            print('缺少数据文件：%s' % p, file=sys.stderr)
            return 1
    count, rows, cols, pixels = read_idx_images(img)
    lcount, labels = read_idx_labels(lab)
    if lcount != count:
        raise RuntimeError('图像数与标签数不一致：%d / %d' % (count, lcount))

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, 'mnist_test.bin')
    with open(path, 'wb') as f:
        f.write(struct.pack('<IIII', MAGIC, count, rows, cols))
        f.write(bytes(labels))
        f.write(bytes(pixels))
    print('wrote %s : %d 张 %dx%d，标签 %d 个，%.2f MB'
          % (path, count, rows, cols, len(labels), os.path.getsize(path) / 1048576.0))
    return 0


if __name__ == '__main__':
    sys.exit(main())
