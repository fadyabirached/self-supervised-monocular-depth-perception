import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (IncludeLaunchDescription, ExecuteProcess,
                             SetEnvironmentVariable, TimerAction)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():

    pkg      = get_package_share_directory('depth_project')
    tb3_gz   = get_package_share_directory('turtlebot3_gazebo')
    world    = os.path.join(pkg, 'worlds', 'columns_world.world')

    # ── TurtleBot3 model (burger has a camera) ──────────────────────────────
    set_model = SetEnvironmentVariable('TURTLEBOT3_MODEL', 'burger')

    # ── Launch Gazebo with our custom world ─────────────────────────────────
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tb3_gz, 'launch', 'turtlebot3_world.launch.py')
        ),
        launch_arguments={
            'world': world,
            'x_pose': '-2.0',
            'y_pose': '-2.0',
            'yaw':    '0.0',
        }.items(),
    )

    # ── depth_node  (starts 5 s after Gazebo to let simulation settle) ──────
    depth_node = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='depth_project',
                executable='depth_node',
                name='depth_node',
                output='screen',
            )
        ]
    )

    # ── robot_controller  (starts 6 s after Gazebo) ─────────────────────────
    robot_controller = TimerAction(
        period=6.0,
        actions=[
            Node(
                package='depth_project',
                executable='robot_controller',
                name='robot_controller',
                output='screen',
            )
        ]
    )

    return LaunchDescription([
        set_model,
        gazebo,
        depth_node,
        robot_controller,
    ])
