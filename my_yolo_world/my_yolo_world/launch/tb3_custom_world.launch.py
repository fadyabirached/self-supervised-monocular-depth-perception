from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction, SetEnvironmentVariable
from launch_ros.actions import Node


def generate_launch_description():
    world = '/home/mhamad/ros2_ws/src/my_yolo_world/worlds/yolo_world.sdf'

    return LaunchDescription([
        SetEnvironmentVariable('TURTLEBOT3_MODEL', 'waffle'),

        ExecuteProcess(
            cmd=['gz', 'sim', '-r', world],
            output='screen'
        ),

        TimerAction(
            period=3.0,
            actions=[
                ExecuteProcess(
                    cmd=['ros2', 'launch', 'turtlebot3_gazebo', 'robot_state_publisher.launch.py'],
                    output='screen'
                )
            ]
        ),

        TimerAction(
            period=7.0,
            actions=[
                Node(
                    package='ros_gz_sim',
                    executable='create',
                    arguments=['-topic', '/robot_description', '-name', 'waffle', '-x', '0', '-y', '0', '-z', '0.3'],
                    output='screen'
                )
            ]
        ),

        TimerAction(
            period=10.0,
            actions=[
                Node(
                    package='ros_gz_bridge',
                    executable='parameter_bridge',
                    arguments=['/model/waffle/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist'],
                    output='screen'
                )
            ]
        ),

        TimerAction(
            period=12.0,
            actions=[
                Node(
                    package='ros_gz_image',
                    executable='image_bridge',
                    arguments=['/camera'],
                    output='screen'
                )
            ]
        ),
    ])
