# ROS 2 Obstacle Avoidance: YOLO vs. Self-Supervised Depth

Comparing two perception approaches for robot obstacle avoidance in a ROS 2 / Gazebo simulation: **YOLO-based object detection** vs. **self-supervised monocular depth estimation**. A TurtleBot3 (burger/waffle) navigates a custom world containing walls and chairs, and each perception branch drives the same steering interface independently so the two approaches can be compared under matched conditions.

![Depth vs YOLO comparison](https://github.com/user-attachments/assets/915f2279-97b2-4213-bc0f-ae87a5d19605)

---

## Table of contents

- [Project logic](#project-logic)
- [Repository structure](#repository-structure)
- [Environment](#environment)
- [Run it in a browser, with no local install](#run-it-in-a-browser-with-no-local-install)
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
  - **photometric reprojection**, warp a source frame into the target frame using predicted depth + relative pose, and penalize the SSIM/L1 mismatch against the real target frame;
  - **edge-aware smoothness regularization** on the predicted disparity;
  - **contrastive consistency** between per-frame embeddings (`contrastive_loss`, InfoNCE-style).
- `train_selfsup_depth.py` runs the training loop and checkpoints `depth_net` + `pose_net` to `checkpoints/selfsup_depth_latest.pth`.

#### Inference logic (`depth_node.py`, decision logic in `steering_logic.py`)
- `DepthNet` predicts a disparity map from the live camera image, converted to metric depth via `disp_to_depth`.
- The depth map is split into three regions, **left / center / right** (`split_regions`), and a percentile of each region's valid depth values is taken (`percentile_of_valid`) and temporally smoothed with a small median filter (`median_filter`) to reduce single-frame noise.
- The robot steers based on which region is more open (`choose_steering`): go straight unless the center is clearly closer than the margin, otherwise turn toward whichever side reads more open.

#### Navigation behavior
- move forward when the center is clear
- turn toward the more open side when an obstacle is close
- no bounding boxes, no object classes, steering is a function of estimated scene geometry only

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
YOLO provides object class and 2D image position, but not direct distance, bounding-box area is used as a rough, non-metric proxy for proximity, which is less principled than the depth branch's actual scene-geometry estimate.

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
│   ├── infer_image.py            # same inference on one saved image: CPU, no ROS (tested)
│   ├── steering_logic.py         # pure-Python region-split / steering math (tested)
│   ├── scan_utils.py             # pure-Python LaserScan helper (tested)
│   ├── robot_controller.py       # /steering_cmd -> /model/waffle/cmd_vel
│   ├── train_selfsup_depth.py    # training loop
│   ├── losses.py                 # SSIM, smoothness, photometric, contrastive losses
│   ├── dataset_sequence.py       # frame-pair dataset for training
│   ├── collect_selfsup.py / auto_collect.py / auto_grid_collect.py  # data collection
│   ├── keyboard_control.py / keyboard_steering.py                  # manual teleop
│   ├── save_images.py            # YOLO training-data capture
│   └── tools/metrics_logger.py   # keyboard-driven per-encounter CSV logger
├── checkpoints/selfsup_depth_latest.pth
├── launch/gazebo.launch.py           # original launch (classic Gazebo, EOL)
├── launch/columns_world.launch.py    # current launch (gz sim / Harmonic)
├── worlds/columns_world.world        # original arena (SDF 1.6, classic)
└── worlds/columns_world.sdf          # same arena + robot, ported to gz sim

my_yolo_world/my_yolo_world/      # custom Gazebo world + world-only launch
├── launch/tb3_custom_world.launch.py
└── worlds/yolo_world.sdf

yolo_nav/yolo_nav/                # YOLO branch nodes
└── yolo_nav/
    ├── yolo_nav_node.py          # detection + steering (ROS node, with live overlay)
    ├── yolo_visual_node.py       # detection-only visualizer (no steering logic)
    └── steering_logic.py         # pure-Python detection / steering math (tested)

tests/                            # pytest unit tests: pure logic above + the trained checkpoint
scripts/                          # setup, run, recording and metrics scripts (see below)
.devcontainer/                    # ROS 2 + Gazebo container with a browser desktop
.github/workflows/ci.yml          # lint + unit tests
```

---

## Environment

This project was built and run with:

| Component | Version |
|---|---|
| ROS 2 distro | **Jazzy Jalisco** |
| Simulator, depth branch | classic **Gazebo** (via `gazebo_ros` + `turtlebot3_gazebo`) |
| Simulator, YOLO branch | **Gazebo Harmonic** / new **`gz sim`** (via `ros_gz_sim`, `ros_gz_bridge`, `ros_gz_image`) |
| Robot | TurtleBot3 (burger for the depth branch, waffle for the YOLO branch) |
| Python | 3.12 |

The two branches deliberately use **two different simulators**, this reflects how the project actually evolved (`depth_project` was built against the TurtleBot3 packages' classic-Gazebo world launcher, `my_yolo_world`/`yolo_nav` against the newer `gz sim`), not a design requirement. Both are declared explicitly in each package's `package.xml`:

- `depth_project/depth_project/package.xml`: `rclpy`, `std_msgs`, `sensor_msgs`, `geometry_msgs`, `turtlebot3_gazebo`, `gazebo_ros`, `launch_ros`
- `my_yolo_world/my_yolo_world/package.xml`: `launch`, `launch_ros`, `ros_gz_sim`, `ros_gz_bridge`, `ros_gz_image`, `turtlebot3_gazebo`, `turtlebot3_description`
- `yolo_nav/yolo_nav/package.xml`: `rclpy`, `sensor_msgs`, `std_msgs`, `python3-opencv`, `ros-jazzy-cv-bridge`

Install ROS 2 Jazzy, classic Gazebo, Gazebo Harmonic (`ros-jazzy-ros-gz`), and the TurtleBot3 packages (`ros-jazzy-turtlebot3*`) via `apt`/`rosdep` per the official [ROS 2 Jazzy](https://docs.ros.org/en/jazzy/) and [TurtleBot3](https://emanual.robotis.com/docs/en/platform/turtlebot3/quick-start/) install guides for your OS.

On top of the ROS 2/Gazebo stack, install the plain pip dependencies used by the perception/training code:

```bash
pip install -r requirements.txt
```

(`torch`, `torchvision`, `opencv-python`, `numpy`, `Pillow`, `ultralytics`, see `requirements.txt` for exact pins and rationale. These are not covered by `package.xml`/`rosdep` since they aren't ROS 2 packages.)

---

## Run it in a browser, with no local install

Standing up ROS 2, Gazebo and TurtleBot3 to look at this project is most of a day's work, and it needs a machine that can render a simulation. `.devcontainer/` removes both requirements: it builds a container with the whole stack preinstalled, runs Gazebo on a virtual X display, and serves that display over noVNC, so the simulation arrives in a browser tab.

Open the repository in **GitHub Codespaces** (Code -> Codespaces -> Create codespace). The container builds, the workspace is linked and compiled by `.devcontainer/post-create.sh`, and the desktop comes up on port 6080. Then:

```bash
scripts/run_depth_gz.sh          # Gazebo + depth model + steering
scripts/record_sim.sh demo.gif   # capture 20 s of it as a GIF
```

Open the forwarded port 6080 (add `/vnc.html` if the browser does not land in the client directly) and the Gazebo window is there, alongside the node's own RGB-next-to-depth window.

Two things worth knowing before you try it:

- **There is no GPU.** Rendering goes through Mesa's `llvmpipe` software rasterizer. This world is four walls, nine cylinders and one robot, which is small enough to work, but expect single-digit frames per second. If Gazebo fails to start at all, `RENDER_ENGINE=ogre scripts/run_depth_gz.sh` falls back to the older render engine, which software-rasterizes more reliably than the default `ogre2`.
- **This runs on a stack the original launch file could not.** `gazebo.launch.py` targets classic Gazebo, which reached end of life in January 2025 and has no package on Ubuntu 24.04 / ROS 2 Jazzy. `columns_world.launch.py` and `worlds/columns_world.sdf` are the same arena ported to `gz sim` (Gazebo Harmonic), which is what Jazzy ships and what the YOLO branch was already using. Same wall and column poses, with the lights, materials and simulator plugins classic used to supply implicitly. See [Running the project](#running-the-project) for both entry points.

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
scripts/run_depth_gz.sh          # gz sim / Gazebo Harmonic, the current stack
scripts/run_depth.sh             # classic Gazebo, the original
```

Two entry points, because the simulator changed underneath the project.

`run_depth_gz.sh` launches `columns_world.launch.py`: `gz sim` with `worlds/columns_world.sdf`, then the `ros_gz` bridges (5s and 6s), then `depth_node` and `robot_controller` (10s and 11s) once camera frames are actually flowing. The world file carries the arena *and* the TurtleBot3 waffle with its camera on `/camera` and DiffDrive on `/model/waffle/cmd_vel`, which is the pairing the two nodes already expect. Paths resolve through `get_package_share_directory`, so it runs from any workspace. Useful arguments: `HEADLESS=true` for no Gazebo window, `RENDER_ENGINE=ogre` on a machine with no GPU.

`run_depth.sh` is the original: `ros2 launch depth_project gazebo.launch.py`, classic Gazebo with `columns_world.world` and a burger TurtleBot3. Kept because it is what the committed checkpoint was trained against, but classic Gazebo is end-of-life and has no Ubuntu 24.04 package, so this will not run on a current install.

Either way `depth_node` finds the checkpoint by looking at `$DEPTH_CHECKPOINT`, then the installed package share, then the legacy `~/ros2_ws/src/depth_project/checkpoints/` path. The repo ships one at `depth_project/depth_project/checkpoints/selfsup_depth_latest.pth`, which step 1 exposes at all of them.

### 3. Run the YOLO branch

```bash
scripts/run_yolo.sh
```

There is no single launch file for this branch, the script composes three pieces:
1. `ros2 launch my_yolo_world tb3_custom_world.launch.py`, `gz sim` with `yolo_world.sdf`, a waffle TurtleBot3, and the `cmd_vel`/camera bridges.
2. `ros2 run depth_project robot_controller`, converts `/steering_cmd` into `/model/waffle/cmd_vel`.
3. `ros2 run yolo_nav yolo_nav_node`, runs YOLOv8n on `/camera` and publishes `/steering_cmd` (downloads `yolov8n.pt` on first run).

Two bugs in that launch file meant piece 1 could never actually have worked, found while wiring up the container in [Run it in a browser](#run-it-in-a-browser-with-no-local-install): the world path was hardcoded to `/home/mhamad/ros2_ws/src/...`, which only exists on the original machine, and the `cmd_vel` bridge advertised plain `geometry_msgs/msg/Twist` while `robot_controller.py` (piece 2, shared with the depth branch) publishes `TwistStamped` on that same topic name. ROS 2 topics have exactly one type, so a `TwistStamped` publisher and a `Twist` bridge on the same topic name never match each other, and no steering command from either branch would have reached `DiffDrive`. Both fixed: the path now resolves through `get_package_share_directory`, and the bridge is `TwistStamped`, matching `columns_world.launch.py`'s. The redundant second robot spawn (`ros2 run ros_gz_sim create -name waffle`, alongside the `<model name="waffle">` the world file already declares) is also gone, for the same reason `columns_world.sdf` never needed one: a world-embedded model with `<static>` unset spawns as a live, controllable entity on its own.

### 4. Run the depth model on a single image, without ROS 2 or Gazebo

Everything above needs a machine that can render the simulation. `infer_image.py` is the same inference path with the ROS parts removed, so the trained model can be run on any saved camera frame, on CPU, with only `torch`, `numpy` and `Pillow` installed:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch   # ~200 MB, no CUDA
pip install numpy Pillow

python depth_project/depth_project/depth_project/infer_image.py frame.png --out side_by_side.png
```

It prints the range of the predicted depth map, the three region depths in metres, and the steering value with the behaviour it corresponds to (`0.0` straight, `+0.8` left, `-0.8` right). `--out` writes the RGB frame next to its magma-coloured depth map, the same pair `depth_node`'s window shows.

Those region depths and that steering value are what `depth_node` would have published for the frame: both come from `steering_logic.py`, imported unchanged, so the offline path cannot drift from what the robot does. The only difference is that a still image has no history for the six-frame median filter to smooth over.

The frame has to come from the world the model was trained in, see [Known limitations](#known-limitations).

### To retrain the depth model

```bash
ros2 run depth_project collect_selfsup   # drive the robot around to collect frames into ~/depth_selfsup_data
ros2 run depth_project train_selfsup_depth
```

---

## Observations

No quantitative benchmark run (timed trials, collision counts, success-rate tables) is checked into this repo, the only artifact tracking that kind of data is `depth_project/tools/metrics_logger.py`, which logs reaction time / success / collision per run to a CSV **outside the repo** (`~/robot_metrics_log.csv`) and was never committed here. So rather than invent numbers, here is an honest qualitative summary based on what the code and the pipeline above actually show:

- **Self-supervised depth** reacts to *geometry*, not object identity: it avoids the walls and the chairs equally well because both simply occlude a region of the depth map. It has no notion of "this is a chair", a labelled ground-truth is never needed at train or inference time.
- **YOLO** reacts to *recognized objects*: it only avoids chairs (the one target class configured in `yolo_nav_node.py`). It would not react to a wall, a person, or any COCO class outside `target_classes`, or to any obstacle the detector misses or mislabels (see the mislabeling note above).
- **Distance estimation**: the depth branch has an explicit (if noisy, self-supervised) metric-ish depth signal per region. The YOLO branch approximates "closeness" purely from bounding-box area, which is a much cruder, non-metric signal and is sensitive to object size/pose.
- **Failure modes observed during development**: YOLO occasionally loses track of the chair while it is partially occluded or at oblique angles (COCO's chair class was not trained on Gazebo-rendered furniture), causing it to fall back to the "searching, spin in place" behavior. The depth branch is more consistent about *something is close ahead* but, being class-agnostic, cannot distinguish an important obstacle from an unimportant one.

**Stated limitation:** the paragraph above is qualitative because no timed trials have been run yet. The tooling to fix that now exists, see the next section, but it needs an actual session at the keyboard to produce numbers, so [Known limitations](#known-limitations) still applies until that CSV is committed.

### Running the depth-vs-YOLO comparison yourself

`metrics_logger.py` turns "which branch is better" from an impression into a CSV. It is a stopwatch, not an automated benchmark: it timestamps when you tell it an obstacle appeared and when the robot's steering first moves off zero afterwards, and asks you to mark whether it collided. A human still has to watch the simulation and call the outcome, because "did the robot successfully avoid that column" is not something the topics alone can tell you.

Three terminals, once the workspace is built:

```bash
scripts/run_depth_gz.sh                          # terminal 1: the simulation
scripts/record_metrics.sh depth                  # terminal 2: press o / c / s / n / q as you watch
```

Mark `o` the moment an obstacle is clearly ahead of the robot, then `s` if it gets past without hitting it or `c` if it collides, then move to the next encounter. The arena is small and enclosed, so a single continuous run produces many encounters, no need to restart the simulation between them. 15 to 20 marked encounters is enough for the rates below to mean something; fewer than 10 and `summarize_metrics.py` will say so.

Repeat for the other branch:

```bash
scripts/run_yolo.sh                               # terminal 1: swap the simulation
scripts/record_metrics.sh yolo                     # terminal 2: same keys
```

Then turn both CSVs into the table this section is missing:

```bash
scripts/summarize_metrics.py ~/depth_metrics.csv ~/yolo_metrics.csv
```

Commit the two CSVs alongside whatever table that prints, and this section stops being qualitative.

### A train/serve mismatch that was silently degrading the depth branch

Being able to run the model outside the simulator turned up a bug the live pipeline had been hiding. `DepthNet` is trained on RGB: `SequenceDataset` opens frames with `Image.open(...).convert('RGB')`. `depth_node.py` converted every incoming `/camera` message to **BGR** (for OpenCV's display window) and then fed *that* to the model, so at inference the network was reading the red channel as blue and vice versa, on inputs whose channel statistics it had never been trained on.

Nothing about this looks broken from the outside. The node starts, the depth map still has structure, and the robot still drives. The cost only shows up when you can score the same frame both ways: on coloured frames the two orderings disagree by roughly **half the mean predicted depth**, and that is more than enough to flip the published steering command from "turn left" to "turn right".

`depth_node.py` now decodes to RGB for the model and converts to BGR only for the OpenCV window. Preprocessing lives in exactly one place (`infer_image.preprocess`), which both the node and the offline script call, so the two paths cannot disagree again, and `tests/test_depth_inference.py` fails if the channel order is ever swapped back.

---

## Tests and CI

Robotics/ROS 2 code that's tightly coupled to `rclpy` nodes, a live Gazebo simulation, or a GPU-loaded model is not meaningfully unit-testable in isolation. The parts of the decision logic that *are* pure Python/NumPy have been factored out of the ROS nodes into standalone modules and covered with `pytest`:

- `depth_project/depth_project/depth_project/steering_logic.py`, depth-map region splitting, percentile-of-valid-depth, temporal median filter, and the left/center/right steering decision (used by `depth_node.py`).
- `depth_project/depth_project/depth_project/scan_utils.py`, LaserScan min-valid-range helper (used by `auto_grid_collect.py`).
- `yolo_nav/yolo_nav/yolo_nav/steering_logic.py`, best-detection selection and the offset/area-based steering decision (used by `yolo_nav_node.py`).

On top of that, `tests/test_depth_inference.py` exercises the **trained model**, not just the logic around it. The architecture lives in one file and the weights in a 14 MB binary, and until `infer_image.py` existed nothing checked that the two still fit together. It loads the committed checkpoint, runs a real forward pass on CPU and asserts the output is the right shape, finite, inside `disp_to_depth`'s clamp range, deterministic, and not the flat sheet an unloaded network produces. It also pins the RGB channel order described in [Observations](#a-trainserve-mismatch-that-was-silently-degrading-the-depth-branch).

Run them locally:

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

The checkpoint tests skip unless `torch` is installed, since the PyPI wheel drags in ~2.5 GB of CUDA packages for a machine that may not have a GPU. To run them, add the CPU build:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
```

`.github/workflows/ci.yml` runs on every push/PR to `main`: `ruff` over the Python this project wrote (not the ament-generated boilerplate, which has its own linters), a `py_compile` pass over every `.py` file, and the full `pytest` suite with CPU torch installed so the checkpoint tests actually execute. It deliberately does **not** try to install ROS 2/Gazebo in CI, that's too heavy for GitHub Actions and would test the CI environment more than the code.

---

## Known limitations

- **No committed quantitative results yet.** The comparison in [Observations](#observations) is qualitative because no timed trials have been logged. [Running the depth-vs-YOLO comparison yourself](#running-the-depth-vs-yolo-comparison-yourself) documents the tooling; it still needs an actual session at the keyboard.
- **The depth model only means anything inside the world it was trained in.** It is self-supervised on frames from one Gazebo world, with no labels and no external scale reference, so it has learned the geometry of *that* scene rather than depth in general. Fed ordinary photographs, every left/center/right region comes back between 0.10 m and 0.28 m against a `disp_to_depth` range of 0.1 to 20 m: it reports that everything is pressed against the lens, and the steering command that falls out of comparing three near-identical numbers is decided by noise. This is why there is no public "upload any image" demo, it would look like it worked while being meaningless. `infer_image.py` is for frames captured from `columns_world.world`.
- **The committed checkpoint is from epoch 1.** `train_selfsup_depth.py` is written to run three epochs and overwrites `checkpoints/selfsup_depth_latest.pth` after each one, and the `epoch` field inside the committed file reads `1`. So the shipped weights are one pass over the collected frames at `lr=1e-4`, not a converged model. That is consistent with how coarse the predicted depths are, and retraining to completion is the first thing to do before quoting any numbers from this branch.
- **Hardcoded absolute workspace paths**: `my_yolo_world/launch/tb3_custom_world.launch.py` still hardcodes its world file under `/home/mhamad/ros2_ws/src/...`, so the YOLO branch depends on `scripts/setup_workspace.sh` symlinking into that exact location. The depth branch no longer does: `columns_world.launch.py` resolves the world through `get_package_share_directory`, and `depth_node.find_checkpoint()` checks `$DEPTH_CHECKPOINT` and the package share before falling back to the old path.
- **`depth_project/launch/custom_world_waffle.launch.py`** duplicates `my_yolo_world/launch/tb3_custom_world.launch.py` (same hardcoded world path, `ExecuteProcess`-based instead of `IncludeLaunchDescription`-based). It isn't used by either run script and looks like an earlier iteration left in place; kept for history rather than deleted.
- **`depth_project/depth_project/yolo_controller.py`** imports a `yolo_msgs` package that isn't declared as a dependency anywhere and isn't registered as a console script in `setup.py`, it's dead/experimental code, not part of either working pipeline.
- **Possible `cmd_vel` topic mismatch on the depth branch**: `robot_controller.py` hardcodes its output topic as `/model/waffle/cmd_vel`, which is the `gz sim` bridge naming pattern created explicitly by `my_yolo_world/launch/tb3_custom_world.launch.py` for the YOLO branch. The depth branch's `gazebo.launch.py` runs a **burger** robot in **classic Gazebo** via `turtlebot3_gazebo`'s own launch file, which does not set up that same bridge/topic, classic TurtleBot3 Gazebo launches typically expose an unnamespaced `/cmd_vel`. This documentation pass read the code but did not run a live simulation to confirm the robot actually moves end-to-end on the depth branch; if `scripts/run_depth.sh` produces steering output but no robot motion, this topic name is the first thing to check.
- **No automated integration testing** of the full ROS 2 + Gazebo pipeline, only the pure decision-logic functions are unit tested (see [Tests and CI](#tests-and-ci)).

---

## License

MIT, see [LICENSE](LICENSE).
