cmake_minimum_required(VERSION 3.20)
project(NeuralNetworkEngineWin LANGUAGES CXX)

# 神经网络模拟引擎（Windows 版）：OpenGL 渲染 + CUDA 计算，不依赖任何第三方库。
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 与宿主编译器一致：可选 CUDA 支持。关掉后引擎走 CPU（对拍与自检仍可用），
# 这时连 CUDA Toolkit 都不需要装：
#   cmake -S . -B <目录> -DNNE_WITH_CUDA=OFF
option(NNE_WITH_CUDA "用 CUDA 加速底层计算" ON)

if(NNE_WITH_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # RTX 50 系（Blackwell, sm_120）。换机器时可在命令行覆盖：
    #   cmake -DCMAKE_CUDA_ARCHITECTURES=<你的架构> ..
    set(CMAKE_CUDA_ARCHITECTURES 120)
  endif()
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

file(GLOB_RECURSE NNE_CORE_SRC CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/core/*.cpp)
file(GLOB_RECURSE NNE_GFX_SRC  CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/gfx/*.cpp)
file(GLOB_RECURSE NNE_UI_SRC   CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/ui/*.cpp)

add_library(nne_core STATIC ${NNE_CORE_SRC})
target_include_directories(nne_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/core)
if(MSVC)
  target_compile_options(nne_core PRIVATE /utf-8 /W3)
endif()

if(NNE_WITH_CUDA)
  file(GLOB_RECURSE NNE_GPU_SRC CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/*.cpp
                                                  ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/*.cu)
else()
  # 只编宿主侧（src/gpu/*.cpp）：没有 .cu 时后端函数表恒为空，全部算子走 core::Ops，
  # 语义与有显卡时完全一致（见 src/gpu/gpu_host.cpp 顶部说明）。
  # 用通配而不是逐个点名，省得以后新增 .cpp 时忘了加进来（曾因此链接报 LNK2019）。
  file(GLOB NNE_GPU_SRC CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/*.cpp)
endif()

add_library(nne_gpu STATIC ${NNE_GPU_SRC})
target_include_directories(nne_gpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu)
target_link_libraries(nne_gpu PUBLIC nne_core)
if(NNE_WITH_CUDA)
  target_link_libraries(nne_gpu PUBLIC CUDA::cudart)
  set_target_properties(nne_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
endif()
if(MSVC)
  # 关键：/utf-8 与 /W3 都不能直接丢给 nvcc（会被当成输入文件），要经 -Xcompiler 转交
  target_compile_options(nne_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/utf-8 /W3>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8 -Xcompiler=/W3>)
endif()

add_library(nne_gfx STATIC ${NNE_GFX_SRC})
target_include_directories(nne_gfx PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/gfx)
# 只用系统自带的 opengl32 / gdi32 / user32，不带任何第三方渲染库
target_link_libraries(nne_gfx PUBLIC opengl32 gdi32 user32 nne_core)
if(MSVC)
  target_compile_options(nne_gfx PRIVATE /utf-8 /W3)
endif()

# 图形界面程序：双击就能用
add_executable(nne WIN32 src/main.cpp ${NNE_UI_SRC})
target_include_directories(nne PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/ui)
target_link_libraries(nne PRIVATE nne_core nne_gpu nne_gfx opengl32 gdi32 user32)
if(MSVC)
  target_compile_options(nne PRIVATE /utf-8 /W3)
  set_target_properties(nne PROPERTIES LINK_FLAGS "/ENTRY:mainCRTStartup")
endif()

# 控制台程序：离屏渲染 + 无窗口自检，验证时不占用屏幕
add_executable(nne_cli src/cli.cpp ${NNE_UI_SRC})
target_include_directories(nne_cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/ui)
target_compile_definitions(nne_cli PRIVATE
  NNE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(nne_cli PRIVATE nne_core nne_gpu nne_gfx opengl32 gdi32 user32)
if(MSVC)
  target_compile_options(nne_cli PRIVATE /utf-8 /W3)
endif()

# 断言测试程序：全部离线对拍与自检
file(GLOB_RECURSE NNE_TEST_SRC CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/tests/*.cpp)
add_executable(nne_tests ${NNE_TEST_SRC})
target_include_directories(nne_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/tests)
# 自检要按绝对路径找 assets 与对拍语料，编译期写死源码目录
target_compile_definitions(nne_tests PRIVATE
  NNE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(nne_tests PRIVATE nne_core nne_gpu nne_gfx opengl32 gdi32 user32)
if(MSVC)
  target_compile_options(nne_tests PRIVATE /utf-8 /W3)
endif()

# 资源目录（示例数字与预训练参数随包发布）
add_custom_command(TARGET nne POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${CMAKE_CURRENT_SOURCE_DIR}/assets $<TARGET_FILE_DIR:nne>/assets)
add_custom_command(TARGET nne_cli POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${CMAKE_CURRENT_SOURCE_DIR}/assets $<TARGET_FILE_DIR:nne_cli>/assets)
add_custom_command(TARGET nne_tests POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${CMAKE_CURRENT_SOURCE_DIR}/assets $<TARGET_FILE_DIR:nne_tests>/assets)
