from setuptools import setup

package_name = 'yolo_nav'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mhamad',
    maintainer_email='mhamad@example.com',
    description='YOLO chair navigation nodes',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'yolo_nav_node = yolo_nav.yolo_nav_node:main',
            'yolo_visual_node = yolo_nav.yolo_visual_node:main',
        ],
    },
)
