cmake_minimum_required(VERSION 3.16)

set(OpenVINO_DIR "/opt/intel/openvino_2024.6.0/runtime/cmake")

find_package(OpenVINO REQUIRED COMPONENTS Runtime)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS
    core
    imgproc
    highgui
    imgcodecs
    dnn
)


# 加入 TensorRT 子目录
add_subdirectory(trt)

# detector 静态库
add_library(detector STATIC
    yolo12.cpp
    KFS.cpp
    wepend.cpp
    lo_shu_grid.cpp
    yolo_ls.cpp
    infor.cpp
    
)

# detector 头文件路径
target_include_directories(detector
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}
)

target_link_libraries(detector
    PUBLIC
        Eigen3::Eigen
        openvino::runtime
        detector_trt
        ${OpenCV_LIBS}
    PRIVATE
        tools
        realsense_lib
)

