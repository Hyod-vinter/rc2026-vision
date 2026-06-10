add_subdirectory(serial)

add_library(Com STATIC
    Com.cpp
)

target_include_directories(Com
    PUBLIC
        ${PROJECT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/serial/include
)

target_link_libraries(Com
    yaml-cpp
    serial
    tools
)

find_package(ament_cmake QUIET)
find_package(rclcpp QUIET)
find_package(std_msgs QUIET)
find_package(rosidl_typesupport_cpp QUIET)
find_package(point_msgs QUIET)

if(NOT ament_cmake_FOUND OR NOT rclcpp_FOUND OR NOT std_msgs_FOUND OR NOT rosidl_typesupport_cpp_FOUND OR NOT point_msgs_FOUND)
    message(WARNING "ROS2 not found, skipping ROS2 specific code.")
else()
    message(STATUS "ROS2 environment found, compiling ROS2-related code.")

    target_sources(Com PRIVATE ROS/receive_ROS.cpp)

    ament_target_dependencies(Com
        rclcpp
        std_msgs
        point_msgs
    )
endif()
