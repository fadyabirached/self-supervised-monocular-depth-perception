"""YOLO branch world launch: gz sim, no hardcoded workspace path.

Two real bugs fixed here, found while wiring up the metrics comparison
between this branch and the depth branch:

1. The world path was hardcoded to /home/mhamad/ros2_ws/src/..., a path
   that only exists on the original author's machine. It now resolves
   through get_package_share_directory, so this runs from any workspace,
   including a container where the user is root rather than mhamad.

2. The cmd_vel bridge advertised plain geometry_msgs/msg/Twist, but
   robot_controller.py (the node run_yolo.sh pairs this world with)
   publishes geometry_msgs/msg/TwistStamped on the same topic name. ROS 2
   topics have exactly one type; a TwistStamped publisher and a Twist
   bridge on /model/waffle/cmd_vel do not match each other, so no
   steering command from either branch ever reached DiffDrive. Fixed to
   TwistStamped, matching robot_controller.py and depth_project's own
   columns_world.launch.py.

Also dropped: the separate turtlebot3_gazebo robot_state_publisher launch
and the `ros2 run ros_gz_sim create -name waffle` spawn. yolo_world.sdf
already declares a <model name="waffle"> directly in the world (verified:
exactly one), so that spawn step was requesting a second entity under the
same name as the one gz sim loads at startup, a duplicate the simulator
either rejects or shadows. depth_project/worlds/columns_world.sdf uses
the identical robot block with no separate spawn step and is confirmed
working, so the embedded model is relied on here the same way.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('my_yolo_world')
    world = os.path.join(pkg, 'worlds', 'yolo_world.sdf')

    headless = LaunchConfiguration('headless')
    render_engine = LaunchConfiguration('render_engine')
    common = ['gz', 'sim', '-r', '--render-engine', render_engine]

    return LaunchDescription([
        DeclareLaunchArgument(
            'headless', default_value='false',
            description='Run gz sim without its GUI window'),
        DeclareLaunchArgument(
            'render_engine', default_value='ogre2',
            description='gz sim render engine: ogre2 (default) or ogre (software-friendly)'),

        SetEnvironmentVariable('TURTLEBOT3_MODEL', 'waffle'),

        ExecuteProcess(
            cmd=common + [world],
            condition=UnlessCondition(headless),
            output='screen'),
        ExecuteProcess(
            cmd=common + ['-s', world],
            condition=IfCondition(headless),
            output='screen'),

        # TwistStamped, to match robot_controller.py's publisher.
        TimerAction(period=5.0, actions=[
            Node(
                package='ros_gz_bridge',
                executable='parameter_bridge',
                arguments=[
                    '/model/waffle/cmd_vel@geometry_msgs/msg/TwistStamped@gz.msgs.Twist'
                ],
                output='screen'),
        ]),

        TimerAction(period=6.0, actions=[
            Node(
                package='ros_gz_image',
                executable='image_bridge',
                arguments=['/camera'],
                output='screen'),
        ]),
    ])
