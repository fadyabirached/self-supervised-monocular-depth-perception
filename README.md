# ROS 2 Obstacle Avoidance: YOLO vs. Self-Supervised Depth

Comparing two perception approaches for robot obstacle avoidance in a ROS 2 / Gazebo simulation: **YOLO-based object detection** vs. **self-supervised monocular depth estimation**. A TurtleBot3 (burger/waffle) navigates a custom world containing walls and chairs, and each perception branch drives the same steering interface independently so the two approaches can be compared under matched conditions.

![Depth vs YOLO comparison](https://github.com/user-attachments/assets/915f2279-97b2-4213-bc0f-ae87a5d19605)

---

## Table of contents

- [Project logic](#project-logic)
- [Repository structure](#repository-structure)
- [Environment](#environment)
- [Running the project](#running-the-project)
- [Observations](#observations)
- [Tests and CI](#tests-and-ci)
- [Known limitations](#known-limitations)
- [License](#license)

---

## Project logic

### 1. Self-supervised depth

#### Training logic
- RGB image sequences are collected from the robot camera in simulation (`collect_selfsup.py` / `auto_collect.py`).
- No manual depth labels are used.
- `DepthNet` (an encoder-decoder CNN, `depth_project/models/depth_net.py`) and `PoseNet` (`depth_project/models/pose_net.py`) are trained jointly with self-supervised losses (`depth_project/losses.py`):
  - **photometric reprojection** — warp a source frame into the target frame using predicted depth + relative pose, and penalize the SSIM/L1 mismatch against the real target frame;
  - **edge-aware smoothness regularization** on the predicted disparity;
  - **contrastive consistency** between per-frame embeddings (`contrastive_loss`, InfoNCE-style).
- `train_selfsup_depth.py` runs the training loop and checkpoints `depth_net` + `pose_net` to `checkpoints/selfsup_depth_latest.pth`.

#### Inference logic (`depth_node.py`, decision logic in `steering_logic.py`)
- `DepthNet` predicts a disparity map from the live camera image, converted to metric depth via `disp_to_depth`.
- The depth map is split into three regions — **left / center / right** (`split_regions`) — and a percentile of each region's valid depth values is taken (`percentile_of_valid`) and temporally smoothed with a small median filter (`median_filter`) to reduce single-frame noise.
- The robot steers based on which region is more open (`choose_steering`): go straight unless the center is clearly closer than the margin, otherwise turn toward whichever side reads more open.

#### Navigation behavior
- move forward when the center is clear
- turn toward the more open side when an obstacle is close
- no bounding boxes, no object classes — steering is a function of estimated scene geometry only

<img width="1282" height="577" alt="RGB and self-supervised depth map side by side" src="https://github.com/user-attachments/assets/ce856f42-c146-44b3-9b2b-3438efe3cec0" />

---

### 2. YOLO

#### Detection logic (`yolo_nav_node.py`, decision logic in `steering_logic.py`)
- Each camera frame is run through YOLOv8n (`ultralytics`), filtered to a target class (`chair`) above a confidence threshold.
- Detected boxes are converted to `{center_x, area}` candidates; the largest-area detection is treated as the immediate obstacle (`best_detection`).
- Steering/speed is chosen from that single detection's horizontal offset from the image center and its bounding-box area (`choose_steering_and_speed`):
  - no detection → spin in place, searching
  - large area (very close) → stop and turn away
  - off-center → turn toward it while creeping forward
  - centered → drive straight

#### Navigation behavior
- object in center → avoid
- object on the left → turn right
- object on the right → turn left

#### Limitation
YOLO provides object class and 2D image position, but not direct distance — bounding-box area is used as a rough, non-metric proxy for proximity, which is less principled than the depth branch's actual scene-geometry estimate.

![YOLO chair detection in Gazebo](https://github.com/user-attachments/assets/6b97259e-94b9-4ef1-bbc4-bf1c6fb0cbaf)

YOLO does not always correctly label the simulated chair, because YOLOv8n is trained on COCO rather than on Gazebo-rendered objects, so it assigns the closest known class based on visual similarity rather than a chair-specific detector.

---

## Repository structure

Three independent ROS 2 (`ament_python`) packages, each nested one level below its top-level folder (`<pkg>/<pkg>/`, the standard colcon package layout):

```
depth_project/depth_project/      # self-supervised depth branch
├── depth_project/
│   ├── models/                   # DepthNet, PoseNet
│   ├── depth_node.py             # inference + steering (ROS node)
│   ├── steering_logic.py         # pure-Python region-split / steering math (tested)
│   ├── scan_utils.py             # pure-Python LaserScan helper (tested)
│   ├── robot_controller.py       # /steering_cmd -> /model/waffle/cmd_vel
│   ├── train_selfsup_depth.py    # training loop
│   ├── losses.py                 # SSIM, smoothness, photometric, contrastive losses
│   ├── dataset_sequence.py       # frame-pair dataset for training
│   ├── collect_selfsup.py / auto_collect.py / auto_grid_collect.py  # data collection
│   ├── keyboard_control.py / keyboard_steering.py                  # manual teleop
│   ├── save_images.py            # YOLO training-data capture
│   └── tools/metrics_logger.py   # optional run-by-run CSV metrics logger
├── checkpoints/selfsup_depth_latest.pth
├── launch/gazebo.launch.py       # full depth-branch launch (classic Gazebo)
└── worlds/columns_world.world

my_yolo_world/my_yolo_world/      # custom Gazebo world + world-only launch
├── launch/tb3_custom_world.launch.py
└── worlds/yolo_world.sdf

yolo_nav/yolo_nav/                # YOLO branch nodes
└── yolo_nav/
    ├── yolo_nav_node.py          # detection + steering (ROS node, with live overlay)
    ├── yolo_visual_node.py       # detection-only visualizer (no steering logic)
    └── steering_logic.py         # pure-Python detection / steering math (tested)

tests/                            # pytest unit tests for the pure logic above
scripts/                          # setup + run scripts (see below)
.github/workflows/ci.yml          # lint + unit tests
```

---

## Environment

This project was built and run with:

| Component | Version |
|---|---|
| ROS 2 distro | **Jazzy Jalisco** |
| Simulator — depth branch | classic **Gazebo** (via `gazebo_ros` + `turtlebot3_gazebo`) |
| Simulator — YOLO branch | **Gazebo Harmonic** / new **`gz sim`** (via `ros_gz_sim`, `ros_gz_bridge`, `ros_gz_image`) |
| Robot | TurtleBot3 (burger for the depth branch, waffle for the YOLO branch) |
| Python | 3.12 |

The two branches deliberately use **two different simulators** — this reflects how the project actually evolved (`depth_project` was built against the TurtleBot3 packages' classic-Gazebo world launcher, `my_yolo_world`/`yolo_nav` against the newer `gz sim`), not a design requirement. Both are declared explicitly in each package's `package.xml`:

- `depth_project/depth_project/package.xml`: `rclpy`, `std_msgs`, `sensor_msgs`, `geometry_msgs`, `turtlebot3_gazebo`, `gazebo_ros`, `launch_ros`
- `my_yolo_world/my_yolo_world/package.xml`: `launch`, `launch_ros`, `ros_gz_sim`, `ros_gz_bridge`, `ros_gz_image`, `turtlebot3_gazebo`, `turtlebot3_description`
- `yolo_nav/yolo_nav/package.xml`: `rclpy`, `sensor_msgs`, `std_msgs`, `python3-opencv`, `ros-jazzy-cv-bridge`

Install ROS 2 Jazzy, classic Gazebo, Gazebo Harmonic (`ros-jazzy-ros-gz`), and the TurtleBot3 packages (`ros-jazzy-turtlebot3*`) via `apt`/`rosdep` per the official [ROS 2 Jazzy](https://docs.ros.org/en/jazzy/) and [TurtleBot3](https://emanual.robotis.com/docs/en/platform/turtlebot3/quick-start/) install guides for your OS.

On top of the ROS 2/Gazebo stack, install the plain pip dependencies used by the perception/training code:

```bash
pip install -r requirements.txt
```

(`torch`, `torchvision`, `opencv-python`, `numpy`, `Pillow`, `ultralytics` — see `requirements.txt` for exact pins and rationale. These are not covered by `package.xml`/`rosdep` since they aren't ROS 2 packages.)

---

## Running the project

### 1. Build the workspace

The launch files assume the three packages live under a colcon workspace at `~/ros2_ws/src/` (see [Known limitations](#known-limitations)). `scripts/setup_workspace.sh` symlinks them there and builds:

```bash
scripts/setup_workspace.sh
```

Equivalently, by hand:

```bash
mkdir -p ~/ros2_ws/src
ln -s "$(pwd)/depth_project/depth_project"   ~/ros2_ws/src/depth_project
ln -s "$(pwd)/my_yolo_world/my_yolo_world"   ~/ros2_ws/src/my_yolo_world
ln -s "$(pwd)/yolo_nav/yolo_nav"             ~/ros2_ws/src/yolo_nav
cd ~/ros2_ws && colcon build --symlink-install
source install/setup.bash
```

### 2. Run the self-supervised depth branch

```bash
scripts/run_depth.sh
```

This is exactly `ros2 launch depth_project gazebo.launch.py`: it starts classic Gazebo with `columns_world.world` and a burger TurtleBot3, then `depth_node` (5s later) and `robot_controller` (6s later) once the simulation has settled. `depth_node` loads the checkpoint at `~/ros2_ws/src/depth_project/checkpoints/selfsup_depth_latest.pth` — the repo ships one at `depth_project/depth_project/checkpoints/selfsup_depth_latest.pth`, which the symlink from step 1 already exposes at that path.

### 3. Run the YOLO branch

```bash
scripts/run_yolo.sh
```

There is no single launch file for this branch — the script composes three pieces:
1. `ros2 launch my_yolo_world tb3_custom_world.launch.py` — `gz sim` with `yolo_world.sdf`, a waffle TurtleBot3, and the `cmd_vel`/camera bridges.
2. `ros2 run depth_project robot_controller` — converts `/steering_cmd` into `/model/waffle/cmd_vel`.
3. `ros2 run yolo_nav yolo_nav_node` — runs YOLOv8n on `/camera` and publishes `/steering_cmd` (downloads `yolov8n.pt` on first run).

### To retrain the depth model

```bash
ros2 run depth_project collect_selfsup   # drive the robot around to collect frames into ~/depth_selfsup_data
ros2 run depth_project train_selfsup_depth
```

---

## Observations

No quantitative benchmark run (timed trials, collision counts, success-rate tables) is checked into this repo — the only artifact tracking that kind of data is `depth_project/tools/metrics_logger.py`, which logs reaction time / success / collision per run to a CSV **outside the repo** (`~/robot_metrics_log.csv`) and was never committed here. So rather than invent numbers, here is an honest qualitative summary based on what the code and the pipeline above actually show:

- **Self-supervised depth** reacts to *geometry*, not object identity: it avoids the walls and the chairs equally well because both simply occlude a region of the depth map. It has no notion of "this is a chair" — a labelled ground-truth is never needed at train or inference time.
- **YOLO** reacts to *recognized objects*: it only avoids chairs (the one target class configured in `yolo_nav_node.py`). It would not react to a wall, a person, or any COCO class outside `target_classes`, or to any obstacle the detector misses or mislabels (see the mislabeling note above).
- **Distance estimation**: the depth branch has an explicit (if noisy, self-supervised) metric-ish depth signal per region. The YOLO branch approximates "closeness" purely from bounding-box area, which is a much cruder, non-metric signal and is sensitive to object size/pose.
- **Failure modes observed during development**: YOLO occasionally loses track of the chair while it is partially occluded or at oblique angles (COCO's chair class was not trained on Gazebo-rendered furniture), causing it to fall back to the "searching — spin in place" behavior. The depth branch is more consistent about *something is close ahead* but, being class-agnostic, cannot distinguish an important obstacle from an unimportant one.

**Stated limitation / future work:** rerun both branches for a fixed number of trials with `metrics_logger.py` enabled and commit the resulting CSV + a comparison table (success rate, mean reaction time, collision rate) instead of this qualitative summary.

---

## Tests and CI

Robotics/ROS 2 code that's tightly coupled to `rclpy` nodes, a live Gazebo simulation, or a GPU-loaded model is not meaningfully unit-testable in isolation. The parts of the decision logic that *are* pure Python/NumPy have been factored out of the ROS nodes into standalone modules and covered with `pytest`:

- `depth_project/depth_project/depth_project/steering_logic.py` — depth-map region splitting, percentile-of-valid-depth, temporal median filter, and the left/center/right steering decision (used by `depth_node.py`).
- `depth_project/depth_project/depth_project/scan_utils.py` — LaserScan min-valid-range helper (used by `auto_grid_collect.py`).
- `yolo_nav/yolo_nav/yolo_nav/steering_logic.py` — best-detection selection and the offset/area-based steering decision (used by `yolo_nav_node.py`).

Run them locally:

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

`.github/workflows/ci.yml` runs on every push/PR to `main`: `ruff` over the whole repo, a `py_compile` pass over every `.py` file, and the `pytest` suite above. It deliberately does **not** try to install ROS 2/Gazebo in CI — that's too heavy for GitHub Actions and would test the CI environment more than the code.

---

## Known limitations

- **No committed quantitative results** — see [Observations](#observations) above.
- **Hardcoded absolute workspace paths**: `my_yolo_world/launch/tb3_custom_world.launch.py` and `depth_project/depth_project/depth_node.py` hardcode paths under `~/ros2_ws/src/...` rather than resolving them via `get_package_share_directory`/parameters. `scripts/setup_workspace.sh` and `scripts/run_*.sh` work around this by symlinking into that exact location.
- **`depth_project/launch/custom_world_waffle.launch.py`** duplicates `my_yolo_world/launch/tb3_custom_world.launch.py` (same hardcoded world path, `ExecuteProcess`-based instead of `IncludeLaunchDescription`-based). It isn't used by either run script and looks like an earlier iteration left in place; kept for history rather than deleted.
- **`depth_project/depth_project/yolo_controller.py`** imports a `yolo_msgs` package that isn't declared as a dependency anywhere and isn't registered as a console script in `setup.py` — it's dead/experimental code, not part of either working pipeline.
- **Possible `cmd_vel` topic mismatch on the depth branch**: `robot_controller.py` hardcodes its output topic as `/model/waffle/cmd_vel`, which is the `gz sim` bridge naming pattern created explicitly by `my_yolo_world/launch/tb3_custom_world.launch.py` for the YOLO branch. The depth branch's `gazebo.launch.py` runs a **burger** robot in **classic Gazebo** via `turtlebot3_gazebo`'s own launch file, which does not set up that same bridge/topic — classic TurtleBot3 Gazebo launches typically expose an unnamespaced `/cmd_vel`. This documentation pass read the code but did not run a live simulation to confirm the robot actually moves end-to-end on the depth branch; if `scripts/run_depth.sh` produces steering output but no robot motion, this topic name is the first thing to check.
- **No automated integration testing** of the full ROS 2 + Gazebo pipeline — only the pure decision-logic functions are unit tested (see [Tests and CI](#tests-and-ci)).

---

## License

MIT — see [LICENSE](LICENSE).
