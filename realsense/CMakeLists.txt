# 生成一个静态库
add_library(realsense_lib
    realsense.cpp
)

target_include_directories(realsense_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR} 
)
target_link_libraries(realsense_lib PUBLIC
  tools                         # ✅ 你的 tools 子目录一般会生成一个 tools target（按你 tools/CMakeLists 里名字）
  realsense2::realsense2
  ${OpenCV_LIBS}
)