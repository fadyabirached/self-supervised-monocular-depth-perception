# ROS 2 Jazzy + Gazebo Harmonic + a browser-reachable desktop.
#
# The point of this image is that the simulation can be watched without a
# local ROS 2 install and without a GPU: Gazebo draws onto a virtual X
# display, x11vnc exports that display, and noVNC serves it over HTTP, so
# the whole thing arrives in a browser tab on port 6080.
#
# There is no GPU in any free tier, so rendering goes through Mesa's
# llvmpipe software rasterizer. The world here is nine cylinders, four
# walls and one robot, which is small enough for that to work, but expect
# single-digit frames per second.
FROM osrf/ros:jazzy-desktop

ARG DEBIAN_FRONTEND=noninteractive

# Simulator, bridges and the TurtleBot3 packages. ros_gz brings Gazebo
# Harmonic in as a dependency, which is the pairing Jazzy ships; classic
# Gazebo is end-of-life and has no Ubuntu 24.04 package, which is why
# columns_world.sdf exists alongside the original .world file.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-jazzy-ros-gz \
        ros-jazzy-ros-gz-sim \
        ros-jazzy-ros-gz-bridge \
        ros-jazzy-ros-gz-image \
        ros-jazzy-turtlebot3 \
        ros-jazzy-turtlebot3-msgs \
        ros-jazzy-turtlebot3-description \
        ros-jazzy-cv-bridge \
        python3-colcon-common-extensions \
        python3-opencv \
        python3-numpy \
        python3-pil \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Virtual display, VNC export, noVNC web client, a window manager so
# Gazebo's windows behave, and ffmpeg for recording the screen.
RUN apt-get update && apt-get install -y --no-install-recommends \
        xvfb \
        x11vnc \
        novnc \
        websockify \
        openbox \
        x11-utils \
        xterm \
        ffmpeg \
        mesa-utils \
        libgl1-mesa-dri \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch. The default PyPI wheel bundles the CUDA runtime and
# pulls roughly 2.5 GB of nvidia-* packages onto a machine with no GPU.
# opencv, numpy and Pillow deliberately come from apt above so they stay
# the same builds cv_bridge was compiled against.
RUN pip3 install --break-system-packages --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision

# Software rendering. Without these Gazebo tries for hardware GL, finds
# none, and dies on startup rather than falling back.
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV GALLIUM_DRIVER=llvmpipe
ENV DISPLAY=:1
ENV TURTLEBOT3_MODEL=waffle
ENV QT_X11_NO_MITSHM=1

COPY start-desktop.sh /usr/local/bin/start-desktop
RUN chmod +x /usr/local/bin/start-desktop

EXPOSE 6080
