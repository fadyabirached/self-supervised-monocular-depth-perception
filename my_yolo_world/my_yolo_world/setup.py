from setuptools import setup
from glob import glob

package_name = 'my_yolo_world'

setup(
    name=package_name,
    version='0.0.1',
    packages=[],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/worlds', glob('worlds/*.sdf')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mhamad',
    maintainer_email='mhamad@example.com',
    description='Custom Gazebo world for TurtleBot3 + YOLO',
    license='Apache License 2.0',
)
