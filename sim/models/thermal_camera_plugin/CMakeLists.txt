cmake_minimum_required(VERSION 3.16)
project(lunar_thermal_camera_plugin LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(gazebo REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${GAZEBO_INCLUDE_DIRS} ${rclcpp_INCLUDE_DIRS} ${sensor_msgs_INCLUDE_DIRS} ${cv_bridge_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${Eigen3_INCLUDE_DIRS} /opt/ros/humble/include/cv_bridge /opt/ros/humble/include/sensor_msgs)
link_directories(${GAZEBO_LIBRARY_DIRS})
add_definitions(${GAZEBO_CXX_FLAGS})

add_library(lunar_thermal_camera_plugin SHARED thermal_camera_plugin.cpp)
target_link_libraries(lunar_thermal_camera_plugin
    ${GAZEBO_LIBRARIES}
    ${rclcpp_LIBRARIES}
    ${sensor_msgs_LIBRARIES}
    ${cv_bridge_LIBRARIES}
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

install(TARGETS lunar_thermal_camera_plugin
    LIBRARY DESTINATION lib
)
