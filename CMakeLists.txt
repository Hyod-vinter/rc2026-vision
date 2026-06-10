cmake_minimum_required(VERSION 3.16.3)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.3/bin/nvcc" CACHE FILEPATH "CUDA compiler")
set(CUDAToolkit_ROOT "/usr/local/cuda-12.3" CACHE PATH "CUDA root")

project(test LANGUAGES C CXX CUDA)

# =========================
# 基础配置
# =========================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-fconstexpr-ops-limit=1000000000)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "-------------------- CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE} --------------------")

# =========================
# 依赖
# =========================
find_package(OpenCV REQUIRED COMPONENTS
    core
    imgproc
    highgui
    videoio
    imgcodecs
    calib3d
    dnn
    objdetect
)

find_package(realsense2 REQUIRED)
find_package(spdlog REQUIRED)
find_package(fmt REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libs   : ${OpenCV_LIBS}")

# =========================
# 子目录
# =========================
add_subdirectory(tools)
add_subdirectory(usbcamera)
add_subdirectory(realsense)
add_subdirectory(task/detector)
add_subdirectory(Com)
add_subdirectory(task/planner)

# =========================
# 可执行文件
# =========================
add_executable(412 src/412.cpp)
add_executable(detect_test test/detect_test.cpp)
add_executable(usbcamera_test test/usbcamera_test.cpp)
add_executable(wepend_detect_test test/wepend_detect_test.cpp)
add_executable(capture_usb capture/capture_usb.cpp)
add_executable(capture_realsense capture/capture_realsense.cpp)
add_executable(video_rs capture/video_rs.cpp)
add_executable(lo_shu_grid_test test/lo_shu_grid_test.cpp)
add_executable(qr_test test/qr_test.cpp)
add_executable(meilin_test test/meilin_test.cpp)

target_link_libraries(412 PRIVATE
    realsense_lib
    realsense2::realsense2
    usbcamera_lib
    spdlog::spdlog
    fmt::fmt
    Eigen3::Eigen
    tools
    detector
    yaml-cpp
    Com
    opencv_core
    opencv_imgproc
    opencv_highgui
    opencv_videoio
    opencv_imgcodecs
    opencv_calib3d
    opencv_dnn
)

# =========================
# detect_test
# =========================
target_link_libraries(detect_test PRIVATE
    realsense_lib
    realsense2::realsense2
    spdlog::spdlog
    fmt::fmt
    Eigen3::Eigen
    tools
    detector
    yaml-cpp
    opencv_core
    opencv_imgproc
    opencv_highgui
    opencv_videoio
    opencv_imgcodecs
    opencv_calib3d
    opencv_dnn
)

# =========================
# usbcamera_test
# =========================
target_link_libraries(usbcamera_test PRIVATE
    usbcamera_lib
    spdlog::spdlog
    fmt::fmt
    Eigen3::Eigen
    tools
    yaml-cpp
    opencv_core
    opencv_imgproc
    opencv_highgui
    opencv_videoio
    opencv_imgcodecs
    opencv_calib3d
)

# =========================
# wepend_detect_test
# =========================
target_link_libraries(wepend_detect_test PRIVATE
    usbcamera_lib
    spdlog::spdlog
    fmt::fmt
    Eigen3::Eigen
    tools
    yaml-cpp
    detector
    opencv_core
    opencv_imgproc
    opencv_highgui
    opencv_videoio
    opencv_imgcodecs
    opencv_calib3d
    opencv_dnn
)


# =========================
# capture_usb
# =========================
target_link_libraries(capture_usb PRIVATE
    usbcamera_lib
    spdlog::spdlog
    fmt::fmt
    Eigen3::Eigen
    tools
    yaml-cpp
    Com
    opencv_core
    opencv_imgproc
    opencv_highgui
    opencv_videoio
    opencv_imgcodecs
    opencv_calib3d
    opencv_dnn
)

# =========================
# capture_realsense
# =========================
target_link_libraries(capture_realsense PRIVATE
    realsense_lib
    realsense2::realsense2
    spdlog::spdlog
    fmt::fmt
    Eigen3::Eigen
    tools
    yaml-cpp
    Com
    opencv_core
    opencv_imgproc
    opencv_highgui
    opencv_videoio
    opencv_imgcodecs
    opencv_calib3d
    opencv_dnn
)

# =========================
# video_rs
# =========================
target_link_libraries(video_rs PRIVATE
    realsense_lib
    realsense2::realsense2
    spdlog::spdlog
    fmt::fmt
    Eigen3::Eigen
    tools
    yaml-cpp
    Com
    opencv_core
    opencv_imgproc
    opencv_highgui
    opencv_videoio
    opencv_imgcodecs
    opencv_calib3d
    opencv_dnn
)

# =========================
# lo_shu_grid_test
# =========================
target_link_libraries(lo_shu_grid_test PRIVATE
    realsense_lib
    realsense2::realsense2
    usbcamera_lib
    spdlog::spdlog
    fmt::fmt
    Eigen3::Eigen
    tools
    detector
    yaml-cpp
    Com
    opencv_core
    opencv_imgproc
    opencv_highgui
    opencv_videoio
    opencv_imgcodecs
    opencv_calib3d
    opencv_dnn
)

# =========================
# qr_test
# =========================
target_link_libraries(qr_test PRIVATE
    realsense_lib
    realsense2::realsense2
    usbcamera_lib
    spdlog::spdlog
    fmt::fmt
    Eigen3::Eigen
    tools
    detector
    yaml-cpp
    Com
    opencv_core
    opencv_imgproc
    opencv_highgui
    opencv_videoio
    opencv_imgcodecs
    opencv_calib3d
    opencv_dnn
    opencv_objdetect
)


# =========================
# meilin_test
# =========================
target_link_libraries(meilin_test PRIVATE
    realsense_lib
    realsense2::realsense2
    usbcamera_lib
    spdlog::spdlog
    fmt::fmt
    Eigen3::Eigen
    tools
    detector
    planner
    yaml-cpp
    Com
    opencv_core
    opencv_imgproc
    opencv_highgui
    opencv_videoio
    opencv_imgcodecs
    opencv_calib3d
    opencv_dnn
    opencv_objdetect
)



find_package(ament_cmake QUIET)
find_package(rclcpp QUIET)
find_package(std_msgs QUIET)
find_package(rosidl_typesupport_cpp QUIET)
find_package(point_msgs QUIET)

if(ament_cmake_FOUND AND rclcpp_FOUND AND std_msgs_FOUND AND rosidl_typesupport_cpp_FOUND AND point_msgs_FOUND)

    message(STATUS "ROS2 environment found, compiling lidar_test.")

    add_executable(lidar_test test/lidar_test.cpp)
    add_executable(main src/main.cpp)

   target_link_libraries(lidar_test
    realsense_lib
    realsense2::realsense2
    usbcamera_lib
    spdlog::spdlog
    fmt::fmt
    Eigen3::Eigen
    tools
    detector
    yaml-cpp
    Com
    opencv_core
    opencv_imgproc
    opencv_highgui
    opencv_videoio
    opencv_imgcodecs
    opencv_calib3d
    opencv_dnn
)
    # =========================
    # main
    # OpenCV 模块显式写出，避免 DSO missing from command line
    # =========================
    target_link_libraries(main 
    realsense_lib
    realsense2::realsense2
    usbcamera_lib
    spdlog::spdlog
    fmt::fmt
    Eigen3::Eigen
    tools
    detector
    planner
    yaml-cpp
    Com
    ${OpenCV_LIBS}
    )

ament_target_dependencies(lidar_test
    rclcpp
    std_msgs
    point_msgs
)

ament_target_dependencies(main
    rclcpp
    std_msgs
    point_msgs
)

else()
    message(STATUS "ROS2 environment not found, skipping lidar_test.")
endif()

