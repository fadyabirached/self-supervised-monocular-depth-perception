from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    turtlebot3_gazebo_dir = get_package_share_directory('turtlebot3_gazebo')
    ros_gz_sim_dir = get_package_share_directory('ros_gz_sim')

    world = '/home/mhamad/ros2_ws/src/my_yolo_world/worlds/yolo_world.sdf'

    return LaunchDescription([
        SetEnvironmentVariable('TURTLEBOT3_MODEL', 'waffle'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([ros_gz_sim_dir, 'launch', 'gz_sim.launch.py'])
            ),
            launch_arguments={'gz_args': f'-r {world}'}.items(),
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([turtlebot3_gazebo_dir, 'launch', 'robot_state_publisher.launch.py'])
            )
        ),

        Node(
            package='ros_gz_sim',
            executable='create',
            arguments=['-topic', '/robot_description', '-name', 'waffle', '-x', '0', '-y', '0', '-z', '0.3'],
            output='screen',
        ),

        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=['/model/waffle/cmd_vel@geometry_msgs/msg/TwistStamped@gz.msgs.Twist'],
            output='screen',
        ),

        Node(
            package='ros_gz_image',
            executable='image_bridge',
            arguments=['/camera'],
            output='screen',
        ),
    ])
