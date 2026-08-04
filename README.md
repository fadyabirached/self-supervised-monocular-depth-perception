# ROS 2 Obstacle Avoidance: YOLO vs. Self-Supervised Depth

[![CI](https://github.com/fadyabirached/self-supervised-monocular-depth-perception/actions/workflows/ci.yml/badge.svg)](https://github.com/fadyabirached/self-supervised-monocular-depth-perception/actions/workflows/ci.yml)

Comparing two perception approaches for robot obstacle avoidance in a **ROS 2 / Gazebo** simulation: **YOLO-based object detection** vs. **self-supervised monocular depth estimation**. A TurtleBot3 navigates a custom world containing walls and columns, and each perception branch drives the same steering interface independently so the two approaches can be compared under matched conditions.

![Depth vs YOLO comparison](https://github.com/user-attachments/assets/915f2279-97b2-4213-bc0f-ae87a5d19605)

> **Status:** paused pending GPU access. The committed depth checkpoint is one epoch into a three-epoch training run, and the pipeline needs a proper retrain before its steering results are trustworthy (see [Known limitations](#known-limitations)). Everything else, the simulation, both branches, the browser-based container, is working and tested.

---

## Architecture

**Self-supervised depth:** `DepthNet` (encoder-decoder CNN) predicts a disparity map from the camera image with no manual labels, trained via photometric reprojection + edge-aware smoothness + contrastive consistency (`losses.py`). At inference, the depth map is split into left/center/right regions; the robot goes straight while the center reads at least as open as the sides, and turns toward whichever side has more room once the center reads clearly nearer (`steering_logic.choose_steering`).

**YOLO:** Each frame runs through YOLOv8n, filtered to a target class (`chair`). The largest detection is treated as the immediate obstacle; steering comes from its horizontal offset and bounding-box area (`yolo_nav/steering_logic.py`). No detection → spin searching, very close → stop and turn away, off-center → turn while creeping forward, centered → drive straight.

| | Self-supervised depth | YOLO |
|---|---|---|
| Reacts to | scene geometry (any obstacle) | recognized objects only (chairs) |
| Distance signal | metric-ish depth per region | bounding-box area (non-metric proxy) |
| Needs labels | no | pretrained on COCO |

<img width="1282" height="577" alt="RGB and self-supervised depth map side by side" src="https://github.com/user-attachments/assets/ce856f42-c146-44b3-9b2b-3438efe3cec0" />
![YOLO chair detection in Gazebo](https://github.com/user-attachments/assets/6b97259e-94b9-4ef1-bbc4-bf1c6fb0cbaf)

---

## Repository structure

Three ROS 2 (`ament_python`) packages, each nested one level below its top-level folder (`<pkg>/<pkg>/`, the standard colcon layout):

```
depth_project/depth_project/      # self-supervised depth branch
├── depth_project/
│   ├── models/                   # DepthNet, PoseNet
│   ├── depth_node.py             # inference + steering (ROS node)
│   ├── infer_image.py            # same inference on one saved image, CPU, no ROS (tested)
│   ├── steering_logic.py         # pure-Python region-split / steering math (tested)
│   ├── train_selfsup_depth.py    # training loop
│   ├── losses.py                 # SSIM, smoothness, photometric, contrastive losses
│   └── tools/metrics_logger.py   # keyboard-driven per-encounter CSV logger
├── checkpoints/selfsup_depth_latest.pth
├── launch/columns_world.launch.py    # current launch (gz sim / Harmonic)
└── worlds/columns_world.sdf          # arena + robot

my_yolo_world/my_yolo_world/      # custom Gazebo world + world-only launch
yolo_nav/yolo_nav/                # YOLO detection + steering nodes (tested)

tests/                            # pytest: pure logic + the trained checkpoint
scripts/                          # setup, run, recording and metrics scripts
.devcontainer/                    # ROS 2 + Gazebo container with a browser desktop
.github/workflows/ci.yml          # lint + unit tests
```

---

## Setup and usage

### Option A: Run it in a browser, no local install

Standing up ROS 2 + Gazebo locally needs a machine that can render a simulation. `.devcontainer/` does it instead: open the repo in **GitHub Codespaces**, and once it builds:

```bash
scripts/setup_workspace.sh
start-desktop
scripts/run_depth_gz.sh          # or scripts/run_yolo.sh
scripts/record_sim.sh demo.gif   # capture it
```

Open forwarded port 6080 (`/vnc.html` if it doesn't land there directly). There's no GPU, so rendering is software (Mesa `llvmpipe`); `RENDER_ENGINE=ogre` is a fallback if Gazebo won't start.

### Option B: Run locally

Requires ROS 2 Jazzy, Gazebo Harmonic (`ros-jazzy-ros-gz`), and the TurtleBot3 packages, installed per the official [ROS 2 Jazzy](https://docs.ros.org/en/jazzy/) and [TurtleBot3](https://emanual.robotis.com/docs/en/platform/turtlebot3/quick-start/) guides, plus `pip install -r requirements.txt`.

```bash
scripts/setup_workspace.sh   # symlinks the 3 packages into ~/ros2_ws and builds
scripts/run_depth_gz.sh      # depth branch
scripts/run_yolo.sh          # YOLO branch
```

### Option C: Run the depth model on one image, no ROS or Gazebo

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch numpy Pillow
python depth_project/depth_project/depth_project/infer_image.py frame.png --out side_by_side.png
```

Same decision logic as `depth_node.py`, imported unchanged, so it can't drift from what the robot does. The frame needs to come from `columns_world`, see [Known limitations](#known-limitations).

### Comparing the two branches

`scripts/record_metrics.sh <depth|yolo>` logs obstacle-encounter outcomes (o/c/s/n keys) to a CSV while you watch either branch run; `scripts/summarize_metrics.py` turns one or two of those CSVs into a success-rate/reaction-time table. No trials have been logged yet, this is what [Known limitations](#known-limitations) means by "no quantitative results."

---

## Testing & CI

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Covers the pure decision logic for both branches (region-splitting, steering math) and, in `test_depth_inference.py`, the **trained checkpoint itself**: a real CPU forward pass asserting correct shape, range, and determinism, so the architecture and the saved weights can't silently drift apart. Checkpoint tests skip unless torch is installed (`pip install --index-url https://download.pytorch.org/whl/cpu torch`, to avoid the ~2.5 GB CUDA wheel).

GitHub Actions (`.github/workflows/ci.yml`) runs ruff + the full pytest suite on every push/PR. It doesn't install ROS 2/Gazebo, that's too heavy for CI and would test the CI environment more than the code.

---

## Known limitations

- **The committed checkpoint is undertrained.** `train_selfsup_depth.py` runs three epochs; the shipped checkpoint is from epoch 1. Retraining to completion, on a GPU, is the next step before the depth branch's steering is worth judging.
- **The depth model only means anything inside the world it was trained in.** Fed an ordinary photograph, every region reads near-identical and close to the lens floor value, no scale reference, no generalization. `infer_image.py` is for `columns_world` frames only.
- **No committed quantitative depth-vs-YOLO comparison.** The tooling exists (`record_metrics.sh` / `summarize_metrics.py`); no trial session has been run yet.
- **`my_yolo_world`'s launch still hardcodes its world path** under `/home/mhamad/ros2_ws/src/...`, so the YOLO branch depends on `scripts/setup_workspace.sh`'s default `~/ros2_ws` layout. The depth branch resolves paths dynamically instead.
- **No automated integration testing** of the full ROS 2 + Gazebo pipeline; only the pure decision-logic functions and the checkpoint forward pass are unit tested.

---

## License

MIT, see [LICENSE](LICENSE).
