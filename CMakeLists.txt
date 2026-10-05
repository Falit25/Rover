cmake_minimum_required(VERSION 3.16)
project(lunar_slam VERSION 0.1.0 LANGUAGES CXX)

if (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Werror=return-type)
endif()

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(nav2_core REQUIRED)
find_package(nav2_msgs REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(gazebo REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Core SLAM Library
add_library(lunar_slam_core
    math_utils.cpp
    traversability_mapper.cpp
    thermal_processing.cpp
    path_planner.cpp
    thermal_inertial_odometry.cpp
    risk_aware_planner.cpp
    sensor_fusion.cpp
    hazard_detector.cpp
    mobility_risk_model.cpp
)

ament_target_dependencies(lunar_slam_core
    rclcpp sensor_msgs nav_msgs tf2_ros geometry_msgs
    tf2_geometry_msgs cv_bridge image_transport nav2_core nav2_msgs
    pcl_ros pcl_conversions
)

target_link_libraries(lunar_slam_core
    PUBLIC
        Eigen3::Eigen
        ${PCL_LIBRARIES}
        ${OpenCV_LIBS}
        ${cv_bridge_LIBRARIES}
        ${image_transport_LIBRARIES}
)

target_include_directories(lunar_slam_core PUBLIC
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${cv_bridge_INCLUDE_DIRS}
    ${image_transport_INCLUDE_DIRS}
)

add_definitions(${PCL_DEFINITIONS})

# Component library for all composable nodes
add_library(lunar_slam_nodes SHARED
    sensor_fusion_node.cpp
    thermal_inertial_odometry_node.cpp
    hazard_detector_node.cpp
    risk_aware_planner_node.cpp
    local_planner_node.cpp
    exploration_manager_node.cpp
    lidar_odometry_node.cpp
    mobility_dataset_collector.cpp
)

ament_target_dependencies(lunar_slam_nodes
    rclcpp
    rclcpp_components
    sensor_msgs
    nav_msgs
    geometry_msgs
    tf2_ros
    tf2_geometry_msgs
    cv_bridge
    image_transport
    pcl_ros
    pcl_conversions
    nav2_core
    nav2_msgs
)

target_link_libraries(lunar_slam_nodes PRIVATE lunar_slam_core)

rclcpp_components_register_nodes(lunar_slam_nodes
    "lunar_slam::SensorFusionNode"
    "lunar_slam::ThermalInertialOdometryNode"
    "lunar_slam::HazardDetectorNode"
    "lunar_slam::RiskAwarePlannerNode"
    "lunar_slam::LocalPlannerNode"
    "lunar_slam::ExplorationManagerNode"
    "lunar_slam::LidarOdometryNode"
    "lunar_slam::MobilityDatasetCollector"
)

# Thermal camera Gazebo plugin
find_package(gazebo REQUIRED)
include_directories(${GAZEBO_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})
add_definitions(${GAZEBO_CXX_FLAGS})

add_library(lunar_thermal_camera_plugin SHARED
    sim/models/thermal_camera_plugin/thermal_camera_plugin.cpp
)

ament_target_dependencies(lunar_thermal_camera_plugin
    rclcpp
    sensor_msgs
    cv_bridge
)

target_link_libraries(lunar_thermal_camera_plugin
    ${GAZEBO_LIBRARIES}
    ${rclcpp_LIBRARIES}
    ${sensor_msgs_LIBRARIES}
    ${cv_bridge_LIBRARIES}
    ${OpenCV_LIBS}
)

install(TARGETS
    lunar_slam_core
    lunar_slam_nodes
    lunar_thermal_camera_plugin
    DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
    DESTINATION include
)

install(DIRECTORY config/
    DESTINATION share/${PROJECT_NAME}/config
)

install(DIRECTORY launch/
    DESTINATION share/${PROJECT_NAME}/launch
)

install(DIRECTORY sim/
    DESTINATION share/${PROJECT_NAME}/sim
)

install(DIRECTORY rviz/
    DESTINATION share/${PROJECT_NAME}/rviz
)

install(DIRECTORY media/
    DESTINATION share/${PROJECT_NAME}/media
)

ament_package()