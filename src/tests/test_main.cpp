/* 断言测试入口：nne_tests [关键字] */
#include <cstdio>

#include "test_util.h"

int main(int argc, char** argv) {
  if (argc > 1) {
    test::setFilter(argv[1]);
    printf("只跑名字里含「%s」的套件\n", argv[1]);
  }
  printf("共登记 %d 个套件\n", test::suiteCount());
  int fails = test::runAll();
  return fails == 0 ? 0 : 1;
}
