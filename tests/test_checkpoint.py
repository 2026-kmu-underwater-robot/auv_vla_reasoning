import json
import subprocess
import sys

import pytest

from auv_vla_reasoning.checkpoint import inspect_checkpoint
from auv_vla_reasoning.controls import STATE_DIMS


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "model"
    (path / "experiment_cfg").mkdir(parents=True)
    metadata = {
        "new_embodiment": {
            "modalities": {
                "state": {key: {"shape": [width]} for key, width in STATE_DIMS.items()},
                "action": {"motion": {"shape": [4]}},
                "video": {"ego": {}, "buoy_release": {}},
            },
            "statistics": {"action": {"motion": {"min": [-1] * 4, "max": [1] * 4}}},
        }
    }
    (path / "experiment_cfg/metadata.json").write_text(json.dumps(metadata))
    (path / "config.json").write_text(json.dumps({"action_head_cfg": {"target_loss_weight": 0}}))
    # Metadata tests never deserialize or execute these placeholder weights.
    (path / "model.safetensors").write_bytes(b"test placeholder")
    return path


@pytest.fixture
def settings():
    return {
        "schema": 1,
        "data_config": "kmu26_auv_real",
        "data_source": "real",
        "use_sim_time": False,
        "neutral_pwm": 1500,
        "pwm_span": 300,
        "expected_mode": "STABILIZE",
        "action_channels": [5, 6, 3, 4],
        "fps": 10,
    }


@pytest.fixture
def runtime(tmp_path, settings):
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(settings))
    return path


def test_legacy_requires_reviewed_runtime(checkpoint, runtime):
    with pytest.raises(ValueError, match="runtime_config"):
        inspect_checkpoint(checkpoint)
    result = inspect_checkpoint(checkpoint, runtime_config=runtime)
    assert result["pwm_span"] == 300
    assert result["data_config"] == "kmu26_auv_real"
    assert result["data_source"] == "real"
    assert str(checkpoint) not in json.dumps(result)


@pytest.mark.parametrize("value", [None, 1, -1, float("nan")])
def test_cap_must_be_persisted_off(checkpoint, runtime, value):
    (checkpoint / "config.json").write_text(
        json.dumps({"action_head_cfg": {"target_loss_weight": value}})
    )
    with pytest.raises(ValueError, match="CAP off"):
        inspect_checkpoint(checkpoint, runtime_config=runtime)


@pytest.mark.parametrize(
    "key,value",
    [
        ("data_source", "simulation"),
        ("use_sim_time", True),
        ("pwm_span", 600),
        ("neutral_pwm", float("nan")),
        ("action_channels", [1, 2, 3, 4]),
        ("fps", 30),
        ("expected_mode", ""),
        ("data_config", "kmu26_auv_real_v2"),
    ],
)
def test_invalid_runtime_rejected(checkpoint, runtime, settings, key, value):
    settings[key] = value
    runtime.write_text(json.dumps(settings))
    with pytest.raises(ValueError):
        inspect_checkpoint(checkpoint, runtime_config=runtime)


def test_simulation_checkpoint_rejected_even_with_real_runtime(checkpoint, runtime):
    (checkpoint / "experiment_cfg/kmu26_transfer.json").write_text("{}")
    with pytest.raises(ValueError, match="Simulation"):
        inspect_checkpoint(checkpoint, runtime_config=runtime)


def test_v2_uses_saved_settings_and_rejects_wrong_preprocessing(checkpoint, settings, runtime):
    settings["data_config"] = "kmu26_auv_real_v2"
    settings["target_loss_weight"] = 0
    del settings["data_source"], settings["use_sim_time"]
    config = {"action_head_cfg": {"target_loss_weight": 0}, "kmu26_training": settings}
    (checkpoint / "config.json").write_text(json.dumps(config))
    result = inspect_checkpoint(checkpoint, "kmu26_auv_real_v2")
    assert result["pwm_span"] == 300
    assert result["use_sim_time"] is False
    with pytest.raises(ValueError, match="select kmu26_auv_real_v2"):
        inspect_checkpoint(checkpoint, runtime_config=runtime)
    settings.update(data_source="real", use_sim_time=False, pwm_span=400)
    runtime.write_text(json.dumps(settings))
    with pytest.raises(ValueError, match="differ"):
        inspect_checkpoint(checkpoint, "kmu26_auv_real_v2", runtime)


def test_legacy_does_not_become_v2_by_selecting_it(checkpoint, runtime):
    with pytest.raises(ValueError, match="original saved"):
        inspect_checkpoint(checkpoint, "kmu26_auv_real_v2", runtime)


@pytest.mark.parametrize("mutation", ["state", "action", "camera", "stats"])
def test_checkpoint_modalities_checked(checkpoint, runtime, mutation):
    path = checkpoint / "experiment_cfg/metadata.json"
    metadata = json.loads(path.read_text())
    body = metadata["new_embodiment"]
    if mutation == "state":
        body["modalities"]["state"]["attitude"]["shape"] = [3]
    elif mutation == "action":
        body["modalities"]["action"]["motion"]["shape"] = [6]
    elif mutation == "camera":
        del body["modalities"]["video"]["buoy_release"]
    else:
        body["statistics"]["action"]["motion"]["min"][0] = float("nan")
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        inspect_checkpoint(checkpoint, runtime_config=runtime)


def test_missing_or_incomplete_weights_rejected(checkpoint, runtime):
    (checkpoint / "model.safetensors").unlink()
    with pytest.raises(ValueError, match="no local model weights"):
        inspect_checkpoint(checkpoint, runtime_config=runtime)
    index = {"weight_map": {"weight": "model-00001-of-00001.safetensors"}}
    (checkpoint / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="shard"):
        inspect_checkpoint(checkpoint, runtime_config=runtime)
    (checkpoint / "model-00001-of-00001.safetensors").write_bytes(b"placeholder")
    assert inspect_checkpoint(checkpoint, runtime_config=runtime)["fps"] == 10


def test_check_only_never_imports_torch(checkpoint, runtime):
    program = (
        "import runpy,sys; "
        "sys.argv=['server','--check_only','--checkpoint',sys.argv[1],'--runtime_config',sys.argv[2]]; "
        "runpy.run_module('auv_vla_reasoning.server',run_name='__main__'); "
        "assert 'torch' not in sys.modules; assert 'gr00t' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", program, str(checkpoint), str(runtime)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["data_config"] == "kmu26_auv_real"


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("root", "action_horizon", 8),
        ("head", "action_horizon", 8),
        ("head", "action_dim", 4),
        ("head", "max_state_dim", 32),
    ],
)
def test_saved_architecture_mismatch_rejected(checkpoint, runtime, section, key, value):
    path = checkpoint / "config.json"
    config = json.loads(path.read_text())
    (config if section == "root" else config["action_head_cfg"])[key] = value
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Model must"):
        inspect_checkpoint(checkpoint, runtime_config=runtime)
