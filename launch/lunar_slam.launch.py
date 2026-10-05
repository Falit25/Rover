from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
import os


def generate_launch_description():
    pkg_share = FindPackageShare('lunar_slam')
    
    params_file = PathJoinSubstitution([pkg_share, 'config', 'params.yaml'])
    rviz_config = PathJoinSubstitution([pkg_share, 'rviz', 'lunar_slam.rviz'])
    world_file = PathJoinSubstitution([pkg_share, 'sim', 'worlds', 'lunar_psr.world'])

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    debug_viz = LaunchConfiguration('debug_viz', default='true')

    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )

    declare_debug_viz = DeclareLaunchArgument(
        'debug_viz', default_value='true',
        description='Enable debug visualization markers'
    )

    # Set Gazebo resource path to find media/heightmaps/crater.png
    gazebo_resource_path = SetEnvironmentVariable(
        name='GAZEBO_RESOURCE_PATH',
        value=[PathJoinSubstitution([pkg_share, 'media'])]
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([FindPackageShare('gazebo_ros'), 'launch', 'gazebo.launch.py'])
        ]),
        launch_arguments={
            'world': world_file,
            'verbose': 'true',
        }.items()
    )

    lunar_slam_container = ComposableNodeContainer(
        name='lunar_slam_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            ComposableNode(
                package='lunar_slam',
                plugin='lunar_slam::LidarOdometryNode',
                name='lidar_odometry_node',
                parameters=[params_file],
                extra_arguments=[{'use_intra_process_comms': True}]
            ),
            ComposableNode(
                package='lunar_slam',
                plugin='lunar_slam::SensorFusionNode',
                name='sensor_fusion_node',
                parameters=[params_file],
                extra_arguments=[{'use_intra_process_comms': True}]
            ),
            ComposableNode(
                package='lunar_slam',
                plugin='lunar_slam::ThermalInertialOdometryNode',
                name='thermal_inertial_odometry_node',
                parameters=[params_file],
                extra_arguments=[{'use_intra_process_comms': True}]
            ),
            ComposableNode(
                package='lunar_slam',
                plugin='lunar_slam::HazardDetectorNode',
                name='hazard_detector_node',
                parameters=[params_file],
                extra_arguments=[{'use_intra_process_comms': True}]
            ),
            ComposableNode(
                package='lunar_slam',
                plugin='lunar_slam::RiskAwarePlannerNode',
                name='risk_aware_planner_node',
                parameters=[params_file],
                extra_arguments=[{'use_intra_process_comms': True}]
            ),
            ComposableNode(
                package='lunar_slam',
                plugin='lunar_slam::LocalPlannerNode',
                name='local_planner_node',
                parameters=[params_file],
                extra_arguments=[{'use_intra_process_comms': True}]
            ),
            ComposableNode(
                package='lunar_slam',
                plugin='lunar_slam::ExplorationManagerNode',
                name='exploration_manager_node',
                parameters=[params_file],
                extra_arguments=[{'use_intra_process_comms': True}]
            ),
        ],
        output='screen',
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(debug_viz)
    )

    return LaunchDescription([
        declare_use_sim_time,
        declare_debug_viz,
        gazebo_resource_path,
        gazebo,
        lunar_slam_container,
        rviz,
    ])