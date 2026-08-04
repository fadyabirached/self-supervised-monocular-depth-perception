"""Run the trained depth model on a saved image, on CPU, with no simulator.

``depth_node.py`` is the real consumer of ``DepthNet``, but it only runs
inside a live ROS 2 + Gazebo session: it needs a ``/camera`` publisher to
subscribe to and an X display to draw on. That makes the trained model
impossible to look at without standing up the whole stack on a machine
that can render the simulation.

This module is the same inference path with the ROS parts removed. It
takes an image file, runs the committed checkpoint on CPU, and reports
exactly what ``DepthNode.image_callback`` would have published for that
frame: the left/center/right region depths and the steering command
``choose_steering`` derives from them. The decision logic itself is
imported from ``steering_logic``, unchanged, so this cannot drift from
what the robot actually does.

Usage::

    python depth_project/depth_project/depth_project/infer_image.py frame.png
    python depth_project/depth_project/depth_project/infer_image.py frame.png --out side_by_side.png

Needs ``torch``, ``numpy`` and ``Pillow``. It does not need ``rclpy``,
OpenCV, CUDA or a GPU.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

if __package__ in (None, ""):
    # Allow `python .../infer_image.py` from anywhere, not just after a
    # colcon build or with PYTHONPATH pointing at the package source root.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depth_project.losses import disp_to_depth
from depth_project.models.depth_net import DepthNet
from depth_project.steering_logic import (
    choose_steering,
    percentile_of_valid,
    split_regions,
)

# The resolution DepthNet was trained at (see SequenceDataset defaults in
# dataset_sequence.py). Feeding it anything else changes the receptive
# field relative to the scene and the depths stop being comparable.
INPUT_HEIGHT = 192
INPUT_WIDTH = 320

DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parents[1] / "checkpoints" / "selfsup_depth_latest.pth"
)

# Percentiles and margin lifted from DepthNode.image_callback so a single
# frame here reports the same numbers the node would publish.
LEFT_PERCENTILE = 35
CENTER_PERCENTILE = 30
RIGHT_PERCENTILE = 35

# Magma control points, sampled at even intervals. Inlined rather than
# pulled from matplotlib or OpenCV: the point of this script is that it
# runs with the smallest possible dependency set.
_MAGMA = np.array(
    [
        [0, 0, 4], [28, 16, 68], [79, 18, 123], [129, 37, 129],
        [181, 54, 122], [229, 80, 100], [251, 135, 97], [254, 194, 135],
        [252, 253, 191],
    ],
    dtype=np.float64,
)


def load_model(checkpoint: str | Path = DEFAULT_CHECKPOINT) -> DepthNet:
    """Load DepthNet from a training checkpoint, in eval mode, on CPU."""
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    state = torch.load(checkpoint, map_location="cpu")
    model = DepthNet()
    model.load_state_dict(state["depth_net"])
    model.eval()
    return model


def preprocess(image: Image.Image) -> torch.Tensor:
    """Turn a PIL image into the tensor DepthNet was trained on.

    Channel order is RGB. That is not a cosmetic detail: ``SequenceDataset``
    trained on ``Image.open(...).convert('RGB')``, so feeding BGR here
    silently scores the model on inputs it never saw. On a coloured frame
    the two orderings disagree by ~50% of the mean predicted depth, which
    is more than enough to flip the steering command.
    """
    resized = image.convert("RGB").resize((INPUT_WIDTH, INPUT_HEIGHT), Image.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def predict_depth(model: DepthNet, image: Image.Image) -> np.ndarray:
    """Predict a metric depth map (H, W) in metres for one image."""
    with torch.no_grad():
        disparity, _embedding = model(preprocess(image))
    return disp_to_depth(disparity).squeeze().numpy()


def region_depths(depth: np.ndarray) -> tuple[float, float, float]:
    """Left/center/right region depths, as DepthNode measures them.

    The node also runs each value through a median filter over the last
    six frames. A single still image has no history to smooth over, so
    these are the unsmoothed readings.
    """
    left, center, right = split_regions(depth)
    return (
        percentile_of_valid(left, LEFT_PERCENTILE),
        percentile_of_valid(center, CENTER_PERCENTILE),
        percentile_of_valid(right, RIGHT_PERCENTILE),
    )


def colorize(depth: np.ndarray) -> Image.Image:
    """Render a depth map as a magma image, near = dark, far = bright.

    Normalized per frame, which is what ``depth_node.py`` displays too.
    It makes the structure visible but means brightness is only comparable
    within one image, never across two.
    """
    span = float(depth.max() - depth.min())
    normalized = (depth - depth.min()) / span if span > 0 else np.zeros_like(depth)

    position = normalized * (len(_MAGMA) - 1)
    low = np.clip(np.floor(position).astype(int), 0, len(_MAGMA) - 2)
    blend = (position - low)[..., None]
    rgb = _MAGMA[low] * (1 - blend) + _MAGMA[low + 1] * blend
    return Image.fromarray(rgb.astype(np.uint8))


def side_by_side(image: Image.Image, depth: np.ndarray) -> Image.Image:
    """The RGB frame next to its depth map, as the node's window shows it."""
    rgb = image.convert("RGB").resize((INPUT_WIDTH, INPUT_HEIGHT), Image.BILINEAR)
    canvas = Image.new("RGB", (INPUT_WIDTH * 2, INPUT_HEIGHT))
    canvas.paste(rgb, (0, 0))
    canvas.paste(colorize(depth), (INPUT_WIDTH, 0))
    return canvas


def describe(steering: float) -> str:
    """The steering value in the words the README uses for it."""
    if steering == 0.0:
        return "go straight (center is clearly the closest region)"
    return "turn left" if steering > 0 else "turn right"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("image", help="path to a camera frame (png/jpg)")
    parser.add_argument(
        "--checkpoint", default=str(DEFAULT_CHECKPOINT), help="training checkpoint to load"
    )
    parser.add_argument(
        "--out", default=None, help="write an RGB-next-to-depth png here"
    )
    args = parser.parse_args(argv)

    model = load_model(args.checkpoint)
    image = Image.open(args.image)
    depth = predict_depth(model, image)
    left, center, right = region_depths(depth)
    steering = choose_steering(left, center, right)

    print(f"depth map    {depth.shape[1]}x{depth.shape[0]}, "
          f"{depth.min():.2f} to {depth.max():.2f} m")
    print(f"left  {left:6.2f} m")
    print(f"center{center:6.2f} m")
    print(f"right {right:6.2f} m")
    print(f"steering {steering:+.2f}  ->  {describe(steering)}")

    if args.out:
        side_by_side(image, depth).save(args.out)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
