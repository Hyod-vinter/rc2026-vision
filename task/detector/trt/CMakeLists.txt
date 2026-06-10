cmake_minimum_required(VERSION 3.16)

# 顶层已经启用了 CUDA，这里不要再重新 project
enable_language(CUDA)

# =========================
# 基础编译选项
# =========================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 生成位置无关代码（做库必须开）
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 如果没有指定编译类型，默认用 Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "-------------------- TRT 编译模式: ${CMAKE_BUILD_TYPE} --------------------")

# =========================
# TensorRT 路径配置
# =========================
set(TENSORRT_ROOT "/home/oy/下载/TensorRT-10.8.0.43")
set(TENSORRT_INCLUDE_DIR "${TENSORRT_ROOT}/include")
set(TENSORRT_LIB_DIR     "${TENSORRT_ROOT}/lib")

# =========================
# CUDA / Eigen
# =========================
find_package(CUDAToolkit REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS
    core
    imgproc
    highgui
    imgcodecs
    dnn
)


# 如有需要，可以手动指定 GPU 架构
set(CMAKE_CUDA_ARCHITECTURES 89)

# =========================
# 查找 TensorRT 库
# =========================
find_library(TENSORRT_NVINFER
    NAMES nvinfer
    HINTS ${TENSORRT_LIB_DIR}
    REQUIRED
)

find_library(TENSORRT_NVINFER_PLUGIN
    NAMES nvinfer_plugin
    HINTS ${TENSORRT_LIB_DIR}
    REQUIRED
)

# =========================
# 源文件
# =========================
set(TRT_SOURCES
    src/block.cpp
    src/model.cpp
    src/postprocess.cpp
    src/postprocess.cu
    src/postprocess_loshu.cu
    src/preprocess.cu
    plugin/yololayer.cu
    yolo12_trt.cpp
    yolo11_lstrt.cpp
    yolo12infor_trt.cpp
    yolo12_numtrt.cpp
    wependtrt.cpp
    lighttrt.cpp
)

# 生成静态库 detector_trt
add_library(detector_trt STATIC ${TRT_SOURCES})

# =========================
# 头文件搜索路径
# =========================
target_include_directories(detector_trt
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/plugin
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${PROJECT_SOURCE_DIR}
        ${TENSORRT_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS}
)

# =========================
# 库目录
# =========================
target_link_directories(detector_trt
    PUBLIC
        ${TENSORRT_LIB_DIR}
)

# =========================
# 宏定义
# =========================
target_compile_definitions(detector_trt
    PRIVATE
        USE_TENSORRT
)

# =========================
# 链接库
# =========================
target_link_libraries(detector_trt
    PUBLIC
        CUDA::cudart
        ${TENSORRT_NVINFER}
        ${TENSORRT_NVINFER_PLUGIN}
        Eigen3::Eigen
        ${OpenCV_LIBS}
    PRIVATE
        tools
)

# =========================
# CUDA 相关属性
# =========================
set_target_properties(detector_trt PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    BUILD_RPATH "${TENSORRT_LIB_DIR}"
)

# =========================
# 调试输出
# =========================
message(STATUS "TensorRT include 路径: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT lib 路径    : ${TENSORRT_LIB_DIR}")
message(STATUS "nvinfer             : ${TENSORRT_NVINFER}")
message(STATUS "nvinfer_plugin      : ${TENSORRT_NVINFER_PLUGIN}")
