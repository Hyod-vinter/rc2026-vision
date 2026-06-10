cmake_minimum_required(VERSION 3.16)

find_package(OpenVINO REQUIRED COMPONENTS Runtime)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS
    core
    imgproc
    highgui
    imgcodecs
    dnn
)

# planner 静态库
add_library(planner STATIC
    board.cpp
    meilin.cpp
)


# planner 头文件路径
target_include_directories(planner
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}
)

target_link_libraries(planner
    PUBLIC
        Eigen3::Eigen
        openvino::runtime
        detector
        ${OpenCV_LIBS}
    PRIVATE
        tools
        realsense_lib
)