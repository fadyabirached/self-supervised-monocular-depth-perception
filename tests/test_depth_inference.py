"""Tests that run the committed checkpoint, not just the decision logic.

The other test modules cover the pure-Python steering rules with hand-made
arrays. Those pass whether or not ``DepthNet`` and the trained weights in
``checkpoints/selfsup_depth_latest.pth`` still fit together, which is the
part most likely to break silently: the architecture lives in one file,
the weights in a binary, and nothing else checks that loading one into the
other still works.

These need torch and Pillow. They skip when those are absent so the
lightweight local run (``pip install -r requirements-dev.txt``) still
works; CI installs the CPU-only torch wheel so they actually execute.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
PIL_Image = pytest.importorskip("PIL.Image")

# Imported after the skips above, so a machine without torch reports these
# as skipped instead of erroring out at collection time.
from depth_project.infer_image import (
    DEFAULT_CHECKPOINT,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    colorize,
    load_model,
    predict_depth,
    preprocess,
    region_depths,
    side_by_side,
)
from depth_project.losses import disp_to_depth
from depth_project.steering_logic import percentile_of_valid, split_regions

# The clamp range disp_to_depth maps a sigmoid disparity into.
MIN_DEPTH, MAX_DEPTH = 0.1, 20.0


@pytest.fixture(scope="module")
def model():
    return load_model()


@pytest.fixture
def frame():
    """A frame with obvious structure, so a collapsed output is visible."""
    array = np.zeros((240, 400, 3), dtype=np.uint8)
    array[:120] = 200
    array[120:] = 150
    array[80:180, 150:250] = 40
    return PIL_Image.fromarray(array)


def test_checkpoint_is_committed():
    assert DEFAULT_CHECKPOINT.exists(), (
        f"{DEFAULT_CHECKPOINT} is missing; the trained weights are part of "
        "the repo and depth_node.py cannot start without them"
    )


def test_checkpoint_loads_into_the_architecture(model):
    assert not model.training, "model must be in eval mode: BatchNorm would otherwise use batch statistics"


def test_prediction_has_the_right_shape_and_stays_in_range(model, frame):
    depth = predict_depth(model, frame)

    assert depth.shape == (INPUT_HEIGHT, INPUT_WIDTH)
    assert np.isfinite(depth).all()
    assert MIN_DEPTH - 1e-6 <= depth.min()
    assert depth.max() <= MAX_DEPTH + 1e-6


def test_prediction_is_deterministic(model, frame):
    first = predict_depth(model, frame)
    second = predict_depth(model, frame)

    np.testing.assert_array_equal(first, second)


def test_prediction_is_not_a_constant_map(model, frame):
    """A depth map with no spatial variation means the forward pass is broken.

    This is the cheapest check that the weights actually loaded: an
    untrained or mis-loaded DepthNet tends to output a near-flat sheet.
    """
    depth = predict_depth(model, frame)

    assert depth.std() > 1e-3


def test_preprocess_produces_a_normalized_nchw_tensor(frame):
    tensor = preprocess(frame)

    assert tuple(tensor.shape) == (1, 3, INPUT_HEIGHT, INPUT_WIDTH)
    assert tensor.dtype == torch.float32
    assert 0.0 <= float(tensor.min()) and float(tensor.max()) <= 1.0


def test_preprocess_keeps_rgb_channel_order():
    """Regression test for a train/serve mismatch.

    SequenceDataset trained on ``Image.open(...).convert('RGB')``, so
    inference has to feed RGB too. depth_node.py used to convert every
    incoming frame to BGR for OpenCV and then hand *that* to the model. A
    pure red image must come back with channel 0 hot and channel 2 cold;
    under BGR it would be the other way round.
    """
    red = PIL_Image.fromarray(
        np.tile(np.array([255, 0, 0], dtype=np.uint8), (INPUT_HEIGHT, INPUT_WIDTH, 1))
    )

    tensor = preprocess(red)[0]

    assert float(tensor[0].mean()) == pytest.approx(1.0)
    assert float(tensor[2].mean()) == pytest.approx(0.0)


def test_channel_order_actually_changes_the_prediction(model):
    """And that the mismatch above is not a distinction without a difference."""
    array = np.zeros((INPUT_HEIGHT, INPUT_WIDTH, 3), dtype=np.uint8)
    array[..., 0] = np.linspace(0, 255, INPUT_WIDTH, dtype=np.uint8)
    array[..., 2] = 255 - array[..., 0]

    as_rgb = predict_depth(model, PIL_Image.fromarray(array))
    as_bgr = predict_depth(model, PIL_Image.fromarray(array[..., ::-1].copy()))

    assert np.abs(as_rgb - as_bgr).mean() > 0.01 * as_rgb.mean()


def test_region_depths_match_the_node_pipeline(model, frame):
    """region_depths must equal what depth_node.py computes by hand."""
    depth = predict_depth(model, frame)
    left_roi, center_roi, right_roi = split_regions(depth)

    assert region_depths(depth) == (
        percentile_of_valid(left_roi, 35),
        percentile_of_valid(center_roi, 30),
        percentile_of_valid(right_roi, 35),
    )


def test_disp_to_depth_is_monotonically_decreasing():
    """Larger disparity means nearer, so depth has to fall as disparity rises."""
    disparity = torch.linspace(0.0, 1.0, 50).reshape(1, 1, 1, 50)

    depth = disp_to_depth(disparity).flatten().numpy()

    assert np.all(np.diff(depth) < 0)
    assert depth[-1] == pytest.approx(MIN_DEPTH)
    assert depth[0] == pytest.approx(MAX_DEPTH)


def test_colorize_and_side_by_side_render(model, frame):
    depth = predict_depth(model, frame)

    assert colorize(depth).size == (INPUT_WIDTH, INPUT_HEIGHT)
    assert side_by_side(frame, depth).size == (INPUT_WIDTH * 2, INPUT_HEIGHT)


def test_colorize_survives_a_flat_depth_map():
    """A constant map makes the per-frame normalization divide by zero."""
    assert colorize(np.full((8, 8), 3.0)).size == (8, 8)
