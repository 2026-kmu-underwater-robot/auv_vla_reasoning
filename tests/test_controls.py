import json

import numpy as np
import pytest

from auv_vla_reasoning.controls import (
    STATE_DIMS,
    CommandLimiter,
    motion_chunk,
    release_channels,
    validate_checkpoint,
)


def test_command_order_limits_and_unowned_channels():
    limiter = CommandLimiter(limit=0.3, slew_per_second=1)
    channels = limiter.apply([1, -1, 0.5, -0.5], 0.1)
    assert [channels[i] for i in (4, 5, 2, 3)] == [1530, 1470, 1530, 1470]
    assert all(channels[i] == 65535 for i in (0, 1, 6, 7, 8, 17))
    for _ in range(10):
        channels = limiter.apply([1, -1, 0.5, -0.5], 0.1)
    assert [channels[i] for i in (4, 5, 2, 3)] == [1590, 1410, 1590, 1410]
    release = release_channels()
    assert [release[i] for i in (4, 5, 2, 3)] == [0] * 4
    assert release[8:] == [65535] * 10


@pytest.mark.parametrize(
    "value",
    [
        np.zeros((16, 8)),
        np.full((16, 4), np.nan),
        np.full((16, 4), 1.1),
        np.zeros((2, 16, 4)),
    ],
)
def test_wrong_or_nonfinite_policy_output_is_rejected(value):
    with pytest.raises(ValueError):
        motion_chunk({"action.motion": value})


def test_singleton_batch_is_supported():
    assert motion_chunk({"action.motion": np.zeros((1, 16, 4))}).shape == (16, 4)


def test_delayed_control_tick_cannot_jump_to_target():
    with pytest.raises(ValueError):
        CommandLimiter().apply(np.ones(4), 3)


def test_u0_metadata_cannot_be_used_as_kmu26_policy(tmp_path):
    folder = tmp_path / "experiment_cfg"
    folder.mkdir()
    path = folder / "metadata.json"
    path.write_text(
        json.dumps({"new_embodiment": {"modalities": {"action": {"pwm": {"shape": [8]}}}}})
    )
    with pytest.raises(ValueError, match="fine-tune"):
        validate_checkpoint(str(tmp_path))
    metadata = {
        "modalities": {
            "state": {key: {"shape": [dim]} for key, dim in STATE_DIMS.items()},
            "action": {"motion": {"shape": [4]}},
            "video": {"ego": {"fps": 10}, "buoy_release": {"fps": 10}},
        },
        "statistics": {"action": {"motion": {"min": [-1] * 4, "max": [1] * 4}}},
    }
    path.write_text(json.dumps({"new_embodiment": metadata}))
    validate_checkpoint(str(tmp_path))


@pytest.mark.parametrize("envelope", [list, tuple])
def test_cap_free_u0_response_is_supported(envelope):
    response = envelope([{"action.motion": np.zeros((16, 4))}, None])
    assert motion_chunk(response).shape == (16, 4)


@pytest.mark.parametrize("response", [None, [], [{}, [1, 2, 3]], [None, None]])
def test_invalid_u0_response_is_rejected(response):
    with pytest.raises(ValueError):
        motion_chunk(response)
