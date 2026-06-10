# 生成一个静态库
add_library(usbcamera_lib
    usbcamera.cpp
)

target_include_directories(usbcamera_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}
)

target_link_libraries(usbcamera_lib PUBLIC
    tools
    ${OpenCV_LIBS}
)
