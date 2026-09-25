/*
 * gfx_png：自己写的 PNG 编码器（不依赖 zlib / libpng / stb）。
 *
 * 做法：
 *  - zlib 流用 deflate 的“存储块”（BTYPE=00，未压缩），每块最多 65535 字节。
 *    图片体积大一点（约等于原始像素），但换来的是几十行就能写完、且被所有解码器接受；
 *  - zlib 头固定 0x78 0x01（CM=8 deflate, CINFO=7, FLEVEL=1），结尾接 adler32；
 *  - 每个 PNG 块（IHDR/IDAT/IEND）算 CRC32（多项式 0xEDB88320）；
 *  - 像素是 RGBA8（color type 6），每行前置一个 filter 字节 0（不过滤）。
 *
 * 写文件在 OneDrive 目录下偶尔会被临时锁住（同步进程占用），这里统一重试 3 次。
 */
#include "gfx_internal.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <system_error>
#include <vector>

#include <filesystem>

namespace gfx {
namespace detail {
namespace {

unsigned int crcTable[256];
bool crcReady = false;

void initCrc() {
  if (crcReady) {
    return;
  }
  for (unsigned int i = 0; i < 256; i++) {
    unsigned int c = i;
    for (int k = 0; k < 8; k++) {
      c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
    }
    crcTable[i] = c;
  }
  crcReady = true;
}

unsigned int crc32Of(const unsigned char* data, size_t n, unsigned int seed = 0xFFFFFFFFu) {
  initCrc();
  unsigned int c = seed;
  for (size_t i = 0; i < n; i++) {
    c = crcTable[(c ^ data[i]) & 0xFFu] ^ (c >> 8);
  }
  return c;
}

unsigned int adler32Of(const unsigned char* data, size_t n) {
  unsigned int a = 1, b = 0;
  for (size_t i = 0; i < n; i++) {
    a += data[i];
    if (a >= 65521u) a -= 65521u;
    b += a;
    if (b >= 65521u) b -= 65521u;
  }
  return (b << 16) | a;
}

void putBE32(std::vector<unsigned char>& v, unsigned int x) {
  v.push_back(static_cast<unsigned char>((x >> 24) & 0xFFu));
  v.push_back(static_cast<unsigned char>((x >> 16) & 0xFFu));
  v.push_back(static_cast<unsigned char>((x >> 8) & 0xFFu));
  v.push_back(static_cast<unsigned char>(x & 0xFFu));
}

void putChunk(std::vector<unsigned char>& out, const char type[4],
              const std::vector<unsigned char>& data) {
  putBE32(out, static_cast<unsigned int>(data.size()));
  size_t start = out.size();
  for (int i = 0; i < 4; i++) {
    out.push_back(static_cast<unsigned char>(type[i]));
  }
  out.insert(out.end(), data.begin(), data.end());
  unsigned int crc = crc32Of(out.data() + start, out.size() - start) ^ 0xFFFFFFFFu;
  putBE32(out, crc);
}

/* 把未压缩数据包成 zlib 流（deflate stored blocks） */
void zipStored(const std::vector<unsigned char>& raw, std::vector<unsigned char>& z) {
  z.push_back(0x78); /* CMF: CM=8, CINFO=7 */
  z.push_back(0x01); /* FLG: 无预设字典，FLEVEL=1，校验和正确 */
  size_t pos = 0;
  if (raw.empty()) {
    z.push_back(0x01); /* BFINAL=1, BTYPE=00 */
    z.push_back(0x00);
    z.push_back(0x00);
    z.push_back(0xFF);
    z.push_back(0xFF);
  }
  while (pos < raw.size()) {
    size_t n = raw.size() - pos;
    if (n > 65535) {
      n = 65535;
    }
    bool last = (pos + n) >= raw.size();
    z.push_back(last ? 0x01 : 0x00);
    z.push_back(static_cast<unsigned char>(n & 0xFF));
    z.push_back(static_cast<unsigned char>((n >> 8) & 0xFF));
    unsigned int inv = static_cast<unsigned int>(~n) & 0xFFFFu;
    z.push_back(static_cast<unsigned char>(inv & 0xFF));
    z.push_back(static_cast<unsigned char>((inv >> 8) & 0xFF));
    z.insert(z.end(), raw.begin() + static_cast<long>(pos),
             raw.begin() + static_cast<long>(pos + n));
    pos += n;
  }
  putBE32(z, adler32Of(raw.data(), raw.size()));
}

bool writeFileRetry(const std::string& path, const std::vector<unsigned char>& bytes,
                    std::string* err) {
  for (int attempt = 1; attempt <= 3; attempt++) {
    std::ofstream f(std::filesystem::u8path(path), std::ios::binary | std::ios::trunc);
    if (f) {
      f.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
      f.flush();
      if (f.good()) {
        f.close();
        return true;
      }
    }
    if (f.is_open()) {
      f.close();
    }
    if (err != nullptr) {
      *err = "写文件失败（第 " + std::to_string(attempt) + " 次）：" + path;
    }
    Sleep(120 * static_cast<DWORD>(attempt)); /* OneDrive 偶尔锁文件，等一会儿再来 */
  }
  return false;
}

}  // namespace

bool ensureParentDir(const std::string& path) {
  std::filesystem::path p = std::filesystem::u8path(path);
  std::filesystem::path dir = p.parent_path();
  if (dir.empty()) {
    return true;
  }
  std::error_code ec;
  for (int attempt = 1; attempt <= 3; attempt++) {
    if (std::filesystem::exists(dir, ec)) {
      return true;
    }
    std::filesystem::create_directories(dir, ec);
    if (!ec && std::filesystem::exists(dir)) {
      return true;
    }
    Sleep(120 * static_cast<DWORD>(attempt));
  }
  return std::filesystem::exists(dir);
}

bool writePng(const std::string& path, int w, int h, const unsigned char* rgba, std::string* err) {
  if (w <= 0 || h <= 0 || rgba == nullptr) {
    if (err != nullptr) {
      *err = "writePng：尺寸或数据非法";
    }
    return false;
  }
  if (!ensureParentDir(path)) {
    if (err != nullptr) {
      *err = "writePng：创建目录失败：" + path;
    }
    return false;
  }

  /* 原始扫描线：每行一个 0 号 filter 字节 + w*4 字节 */
  std::vector<unsigned char> raw;
  raw.reserve(static_cast<size_t>(h) * (1 + static_cast<size_t>(w) * 4));
  for (int y = 0; y < h; y++) {
    raw.push_back(0);
    const unsigned char* row = rgba + static_cast<size_t>(y) * static_cast<size_t>(w) * 4;
    raw.insert(raw.end(), row, row + static_cast<size_t>(w) * 4);
  }

  std::vector<unsigned char> z;
  zipStored(raw, z);

  std::vector<unsigned char> out;
  out.reserve(z.size() + 128);
  static const unsigned char kSig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
  out.insert(out.end(), kSig, kSig + 8);

  std::vector<unsigned char> ihdr;
  putBE32(ihdr, static_cast<unsigned int>(w));
  putBE32(ihdr, static_cast<unsigned int>(h));
  ihdr.push_back(8); /* 位深 */
  ihdr.push_back(6); /* 颜色类型：真彩 + alpha */
  ihdr.push_back(0); /* 压缩方法：deflate */
  ihdr.push_back(0); /* 过滤方法：标准 */
  ihdr.push_back(0); /* 交错：无 */
  putChunk(out, "IHDR", ihdr);
  putChunk(out, "IDAT", z);
  putChunk(out, "IEND", std::vector<unsigned char>());

  return writeFileRetry(path, out, err);
}

}  // namespace detail
}  // namespace gfx
