/* gfx_loader 的实现：用 wglGetProcAddress + GetProcAddress(opengl32) 取入口。 */
#include "gfx_loader.h"

#include <cstdio>

namespace gfxgl {

#define GFXGL_DEF(ret, name, args) PFN_##name name = nullptr;
GFXGL_FUNCS(GFXGL_DEF)
#undef GFXGL_DEF

namespace {
struct Entry {
  void** slot;
  const char* name;
};

std::string g_missing;
int g_loaded = 0;

void* sysLookup(const char* name) {
  /* 先问 WGL（2.0+ 只能这样取），再退回 opengl32.dll（1.1 的老入口）。 */
  void* p = reinterpret_cast<void*>(wglGetProcAddress(name));
  if (p == nullptr || p == reinterpret_cast<void*>(1) || p == reinterpret_cast<void*>(2) ||
      p == reinterpret_cast<void*>(3) || p == reinterpret_cast<void*>(-1)) {
    static HMODULE gl = LoadLibraryW(L"opengl32.dll");
    p = nullptr;
    if (gl != nullptr) {
      p = reinterpret_cast<void*>(GetProcAddress(gl, name));
    }
  }
  return p;
}
}  // namespace

#define GFXGL_ENTRY(ret, name, args) {reinterpret_cast<void**>(&name), #name},
static const Entry kEntries[] = {GFXGL_FUNCS(GFXGL_ENTRY)};
#undef GFXGL_ENTRY

int totalCount() { return static_cast<int>(sizeof(kEntries) / sizeof(kEntries[0])); }

int load(std::string* err) {
  g_missing.clear();
  g_loaded = 0;
  for (size_t i = 0; i < sizeof(kEntries) / sizeof(kEntries[0]); i++) {
    void* p = sysLookup(kEntries[i].name);
    *kEntries[i].slot = p;
    if (p != nullptr) {
      g_loaded++;
    } else if (!g_missing.empty()) {
      g_missing += ", ";
      g_missing += kEntries[i].name;
    } else {
      g_missing = kEntries[i].name;
    }
  }
  if (err != nullptr) {
    *err = g_missing;
  }
  return totalCount() - g_loaded;
}

int loadedCount() { return g_loaded; }

const std::string& missing() { return g_missing; }

}  // namespace gfxgl
