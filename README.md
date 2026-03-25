# ROS 2 Obstacle Avoidance: YOLO vs Self-Supervised Depth

## Project Overview

This project compares two perception approaches for robot obstacle avoidance in a ROS 2 simulation environment: YOLO-based object detection and self-supervised monocular depth estimation. A TurtleBot3 Waffle robot navigates in a custom Gazebo world containing walls and chairs. The objective is to evaluate how each perception method influences navigation performance and obstacle avoidance behavior.

![WhatsApp Image 2026-03-25 at 5 11 52 PM (1)](https://github.com/user-attachments/assets/915f2279-97b2-4213-bc0f-ae87a5d19605)
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
- The depth model predicts a depth map from the camera image.
- The depth map is divided into regions:
  - left
  - center
  - right
- The robot chooses steering based on which region is safer.

#### Navigation behavior
- move forward when center is clear
- turn left or right when an obstacle is close
- avoid walls and chairs using estimated scene depth
  
![WhatsApp Image 2026-03-25 at 5 04 33 PM](https://github.com/user-attachments/assets/5956dee9-022a-4264-bb40-3c2970590099)
---

### 2. YOLO 
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

![WhatsApp Image 2026-03-25 at 5 15 14 PM](https://github.com/user-attachments/assets/6b97259e-94b9-4ef1-bbc4-bf1c6fb0cbaf)
YOLO does not always correctly label the simulated chair because it was trained on the COCO dataset rather than on Gazebo-specific objects, so it assigns the closest known class based on visual similarity.

---

## Environment

The project uses a custom Gazebo world containing:
- four walls
- four office chairs
- TurtleBot3 Waffle robot

World file:
```bash
src/my_yolo_world/worlds/yolo_world.sdf
