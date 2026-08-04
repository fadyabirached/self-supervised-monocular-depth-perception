"""Depth branch on modern gz sim (Gazebo Harmonic), with no absolute paths.

``gazebo.launch.py`` is the original entry point for this branch. It
targets classic Gazebo, which is end-of-life and has no package on Ubuntu
24.04 / ROS 2 Jazzy, and it reaches the robot through
``turtlebot3_gazebo``'s own launch file, whose arguments differ between
distros. Neither is true of this one: it loads
``worlds/columns_world.sdf``, which carries the arena and the robot
itself, so the only external ROS packages needed are the two bridges.

The world path is resolved through ``get_package_share_directory`` rather
than hardcoded under ``~/ros2_ws/src``, so this runs from any colcon
workspace, including the container in ``.devcontainer/``.

Topics, matching what the nodes already expect:

    gz  /camera                 -> ros  /camera                (depth_node reads)
    ros /model/waffle/cmd_vel   -> gz   /model/waffle/cmd_vel   (robot_controller writes)

Start with ``headless:=true`` to run the simulation without the Gazebo
GUI. Sensors still render, so depth_node still gets frames, which is what
a machine with no display needs.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('depth_project')
    world = os.path.join(pkg, 'worlds', 'columns_world.sdf')

    headless = LaunchConfiguration('headless')
    render_engine = LaunchConfiguration('render_engine')

    # -r starts the simulation running rather than paused; -s is the
    # server only, no GUI. The render engine is worth overriding on a
    # machine with no GPU: ogre2 is the default but software-rasterizes
    # poorly, and ogre is the more forgiving fallback.
    common = ['gz', 'sim', '-r', '--render-engine', render_engine]

    return LaunchDescription([
        DeclareLaunchArgument(
            'headless', default_value='false',
            description='Run gz sim without its GUI window'),
        DeclareLaunchArgument(
            'render_engine', default_value='ogre2',
            description='gz sim render engine: ogre2 (default) or ogre (software-friendly)'),

        ExecuteProcess(
            cmd=common + [world],
            condition=UnlessCondition(headless),
            output='screen'),
        ExecuteProcess(
            cmd=common + ['-s', world],
            condition=IfCondition(headless),
            output='screen'),

        # Steering commands out to the simulator. robot_controller
        # publishes TwistStamped, which is what Jazzy standardized on, so
        # that is the ROS type bridged onto gz.msgs.Twist here.
        TimerAction(period=5.0, actions=[
            Node(
                package='ros_gz_bridge',
                executable='parameter_bridge',
                arguments=[
                    '/model/waffle/cmd_vel@geometry_msgs/msg/TwistStamped@gz.msgs.Twist'
                ],
                output='screen'),
        ]),

        # Camera frames in.
        TimerAction(period=6.0, actions=[
            Node(
                package='ros_gz_image',
                executable='image_bridge',
                arguments=['/camera'],
                output='screen'),
        ]),

        # Perception and control, once frames are actually flowing.
        TimerAction(period=10.0, actions=[
            Node(
                package='depth_project',
                executable='depth_node',
                name='depth_node',
                output='screen'),
        ]),
        TimerAction(period=11.0, actions=[
            Node(
                package='depth_project',
                executable='robot_controller',
                name='robot_controller',
                output='screen'),
        ]),
    ])
