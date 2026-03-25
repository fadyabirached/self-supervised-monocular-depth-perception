from setuptools import setup, find_packages
from glob import glob
import os

package_name = 'depth_project'

setup(
    name=package_name,
    version='0.0.2',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mhamad',
    maintainer_email='mhamad@example.com',
    description='Self-supervised monocular depth project',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'robot_controller = depth_project.robot_controller:main',
            'depth_node = depth_project.depth_node:main',
            'keyboard_control = depth_project.keyboard_control:main',
            'keyboard_steering = depth_project.keyboard_steering:main',
            'collect_selfsup = depth_project.collect_selfsup:main',
            'train_selfsup_depth = depth_project.train_selfsup_depth:main',
        ],
    },
)
