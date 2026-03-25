# ROS 2 Obstacle Avoidance: YOLO vs Self-Supervised Depth

## Project Overview

This project compares two perception approaches for robot obstacle avoidance in ROS 2 simulation:

- **YOLO-based object detection**
- **Self-supervised monocular depth estimation**

A TurtleBot3 Waffle robot moves in a custom Gazebo environment containing walls and chairs.  
The goal is to evaluate how each perception method affects robot navigation and obstacle avoidance behavior.

---

## Project Logic

### 1. Self-Supervised Depth Branch
The depth branch is based on a self-supervised monocular depth pipeline.

#### Training logic
- RGB image sequences are collected from the robot camera in simulation.
- No manual depth labels are used.
- The model is trained using self-supervised losses:
  - photometric reconstruction
  - smoothness regularization
  - optional contrastive consistency

#### Inference logic
- The trained depth model predicts a depth map from the camera image.
- The depth map is divided into regions:
  - left
  - center
  - right
- The robot chooses steering based on which region is safer.

#### Navigation behavior
- move forward when center is clear
- turn left or right when an obstacle is close
- avoid walls and chairs using estimated scene depth

---

### 2. YOLO Branch
The YOLO branch uses object detection for obstacle-related navigation.

#### Detection logic
- The camera image is processed by YOLO.
- Detected objects are localized by bounding boxes.
- Their image position is used to generate steering decisions.

#### Navigation behavior
- object in center → avoid
- object on left → turn right
- object on right → turn left

#### Limitation
YOLO provides object class and image position, but not direct distance estimation.

---

## Environment

The project uses a custom Gazebo world containing:
- four walls
- four office chairs
- TurtleBot3 Waffle robot

World file:
```bash
src/my_yolo_world/worlds/yolo_world.sdf
