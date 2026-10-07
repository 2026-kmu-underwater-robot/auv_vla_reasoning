"""Check local real-vehicle model metadata before loading the GPU runtime."""

import json
import math
from pathlib import Path

from .controls import validate_checkpoint

DATA_CONFIGS = ("kmu26_auv_real", "kmu26_auv_real_v2")
SETTINGS_KEYS = (
    "action_channels",
    "fps",
    "neutral_pwm",
    "pwm_span",
    "expected_mode",
)


def _read_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read JSON settings: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _validate_settings(settings: dict, data_config: str, *, runtime: bool) -> None:
    if settings.get("schema") != 1 or settings.get("data_config") != data_config:
        raise ValueError("Settings schema or data_config does not match this model")
    if settings.get("action_channels") != [5, 6, 3, 4] or settings.get("fps") != 10:
        raise ValueError("Expected action_channels [5, 6, 3, 4] and fps 10")
    neutral, span = settings.get("neutral_pwm"), settings.get("pwm_span")
    if any(
        type(value) not in (int, float) or not math.isfinite(value) for value in (neutral, span)
    ):
        raise ValueError("neutral_pwm and pwm_span must be finite numbers")
    if span <= 0 or neutral - span < 1000 or neutral + span > 2000:
        raise ValueError("PWM range must fit within 1000..2000")
    if not isinstance(settings.get("expected_mode"), str) or not settings["expected_mode"].strip():
        raise ValueError("expected_mode must be a non-empty string")
    # v2 metadata is written only by the reviewed-real-data training path. Its
    # original schema omits these two flags; reject contradictory flags if added.
    if runtime or "data_source" in settings or "use_sim_time" in settings:
        if settings.get("data_source") != "real" or settings.get("use_sim_time") is not False:
            raise ValueError("Real inference requires data_source real and use_sim_time false")


def _check_weights(checkpoint: Path) -> None:
    if any((checkpoint / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")):
        return
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = checkpoint / name
        if not index_path.is_file():
            continue
        shards = _read_object(index_path).get("weight_map")
        if not isinstance(shards, dict) or not shards:
            raise ValueError("Checkpoint weight index has no shards")
        for filename in shards.values():
            if not isinstance(filename, str):
                raise ValueError("Invalid checkpoint shard filename")
            shard = (checkpoint / filename).resolve()
            if not shard.is_relative_to(checkpoint) or not shard.is_file():
                raise ValueError("Missing or invalid local checkpoint weight shard")
        return
    raise ValueError("Checkpoint has no local model weights")


def inspect_checkpoint(
    checkpoint: str | Path,
    data_config: str = "kmu26_auv_real",
    runtime_config: str | Path | None = None,
) -> dict:
    """Return reviewed real command settings without importing torch.

    Legacy models require a separate reviewed runtime JSON because their model
    files did not save RC units. v2 models must retain the settings saved by
    training; selecting v2 does not convert a legacy checkpoint.
    """
    if data_config not in DATA_CONFIGS:
        raise ValueError(f"Unsupported data_config: {data_config}")
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_dir():
        raise ValueError("Provide an existing local checkpoint directory")
    if (checkpoint / "experiment_cfg/kmu26_transfer.json").exists():
        raise ValueError("Simulation transfer checkpoints are not accepted by the real server")
    config = _read_object(checkpoint / "config.json")
    action_head = config.get("action_head_cfg")
    if not isinstance(action_head, dict) or action_head.get("target_loss_weight") != 0:
        raise ValueError(
            "config.json must persist action_head_cfg.target_loss_weight = 0 (CAP off)"
        )
    for model_config in (config, action_head):
        if model_config.get("action_horizon", 16) != 16:
            raise ValueError("Model must predict 16 action steps")
        if model_config.get("action_dim", 32) != 32:
            raise ValueError("Model must retain the KMU26 padded action dimension 32")
    if action_head.get("max_state_dim", 64) != 64:
        raise ValueError("Model must retain the KMU26 padded state dimension 64")
    settings = config.get("kmu26_training")
    if data_config == "kmu26_auv_real_v2":
        if not isinstance(settings, dict):
            raise ValueError("v2 requires the original saved kmu26_training settings")
        _validate_settings(settings, data_config, runtime=False)
        if settings.get("target_loss_weight") != 0:
            raise ValueError("v2 training settings must persist target_loss_weight = 0")
    elif settings is not None:
        raise ValueError("Checkpoint contains v2 training settings; select kmu26_auv_real_v2")
    elif runtime_config is None:
        raise ValueError("Legacy checkpoints require an explicitly reviewed --runtime_config JSON")
    if runtime_config is not None:
        reviewed = _read_object(Path(runtime_config).expanduser())
        _validate_settings(reviewed, data_config, runtime=True)
        if settings is not None and any(reviewed[key] != settings[key] for key in SETTINGS_KEYS):
            raise ValueError("Runtime settings differ from the recorded training settings")
        settings = reviewed
    try:
        validate_checkpoint(str(checkpoint), "new_embodiment")
    except (TypeError, AttributeError, KeyError) as exc:
        raise ValueError("Malformed KMU26 checkpoint metadata") from exc
    _check_weights(checkpoint)
    return {
        "schema": 1,
        "model": "auv_vla",
        "data_config": data_config,
        "embodiment_tag": "new_embodiment",
        **{key: settings[key] for key in SETTINGS_KEYS},
        "data_source": "real",
        "use_sim_time": False,
        "action_horizon": 16,
        "action_order": ["surge", "sway", "heave", "yaw"],
        "target_loss_weight": 0,
    }
