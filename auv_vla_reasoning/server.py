"""Serve an existing KMU26 real-vehicle VLA checkpoint over HTTP."""

import argparse
import json
import logging
import os
import sys
import threading
from pathlib import Path
from uuid import uuid4

from .checkpoint import DATA_CONFIGS, inspect_checkpoint
from .controls import STATE_DIMS

TASK_KEY = "annotation.human.action.task_description"


def validate_observation(observation: object) -> dict:
    """Validate one time step of two RGB cameras and 23 physical state values."""
    import numpy as np

    if not isinstance(observation, dict):
        raise ValueError("observation must be an object")
    expected = {
        *(f"state.{key}" for key in STATE_DIMS),
        "video.ego",
        "video.buoy_release",
        TASK_KEY,
    }
    if set(observation) != expected:
        raise ValueError(
            f"Observation keys mismatch; missing={sorted(expected - observation.keys())}, "
            f"unexpected={sorted(observation.keys() - expected)}"
        )
    result = {}
    for name, width in STATE_DIMS.items():
        key = f"state.{name}"
        value = np.asarray(observation[key])
        if (
            value.shape != (1, width)
            or value.dtype.kind not in "fiu"
            or not np.isfinite(value).all()
        ):
            raise ValueError(f"{key} must be finite numeric shape (1, {width})")
        with np.errstate(over="ignore"):
            value = value.astype(np.float32)
        if not np.isfinite(value).all():
            raise ValueError(f"{key} cannot be represented as float32")
        result[key] = value
    if np.any(np.abs(result["state.prev_command"]) > 1):
        raise ValueError("state.prev_command must lie within [-1, 1]")
    if not np.isin(result["state.validity"], (0, 1)).all():
        raise ValueError("state.validity must contain 0 or 1")
    for key in ("video.ego", "video.buoy_release"):
        value = np.asarray(observation[key])
        if (
            value.ndim != 4
            or value.shape[0] != 1
            or value.shape[-1] != 3
            or min(value.shape[1:3]) < 1
        ):
            raise ValueError(f"{key} must have RGB shape (1, H, W, 3)")
        if isinstance(observation[key], np.ndarray) and value.dtype != np.uint8:
            raise ValueError(f"{key} must use uint8 pixels")
        if value.dtype.kind not in "iu" or np.any(value < 0) or np.any(value > 255):
            raise ValueError(f"{key} must contain integer RGB pixels in [0, 255]")
        result[key] = value.astype(np.uint8, copy=False)
    task = observation[TASK_KEY]
    if isinstance(task, np.ndarray):
        task = task.tolist()
    if (
        not isinstance(task, (list, tuple))
        or len(task) != 1
        or not isinstance(task[0], str)
        or not task[0].strip()
    ):
        raise ValueError(f"{TASK_KEY} must contain one non-empty task string")
    result[TASK_KEY] = list(task)
    return result


def validate_action(response: object) -> object:
    """Accept CAP-free normalized actions without silently clipping model errors."""
    import numpy as np

    actions = response
    if isinstance(response, (list, tuple)):
        if len(response) != 2 or response[1] is not None:
            raise ValueError("Model must return CAP-free [actions, null]")
        actions = response[0]
    if not isinstance(actions, dict) or set(actions) != {"action.motion"}:
        raise ValueError("Model must return action.motion only")
    chunk = np.asarray(actions["action.motion"])
    if chunk.shape not in ((16, 4), (1, 16, 4)) or chunk.dtype.kind not in "fiu":
        raise ValueError("Model action.motion must have shape (16, 4) or (1, 16, 4)")
    if not np.isfinite(chunk).all() or np.any(chunk < -1) or np.any(chunk > 1):
        raise ValueError("Model action.motion must be finite and normalized to [-1, 1]")
    return response


def create_app(policy: object, metadata: dict):
    """Create an HTTP app around an injected policy, also usable without weights."""
    import json_numpy
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import Response
    from starlette.concurrency import run_in_threadpool

    app = FastAPI(title="KMU26 real VLA inference", version="0.1.0")
    metadata = {**metadata, "server_instance_id": str(uuid4())}
    inference_lock = threading.Lock()

    def infer(observation):
        if metadata.get("data_config") == "kmu_auv":
            # The ROS contract carries validity for monitoring, but kmu_auv was
            # trained without these four fields.
            observation = {
                key: value for key, value in observation.items() if key != "state.validity"
            }
        with inference_lock:
            action = validate_action(policy.get_action(observation))
            return json_numpy.dumps(action)

    @app.get("/health")
    def health():
        return {
            "status": "healthy",
            "model": "auv_vla",
            "server_instance_id": metadata["server_instance_id"],
        }

    @app.get("/metadata")
    def get_metadata():
        return metadata

    @app.post("/act")
    async def act(request: Request):
        try:
            payload = json_numpy.loads(await request.body())
            if isinstance(payload, dict) and "encoded" in payload:
                if set(payload) != {"encoded"} or not isinstance(payload["encoded"], str):
                    raise ValueError("encoded payload must be the only field and contain JSON text")
                payload = json_numpy.loads(payload["encoded"])
            if not isinstance(payload, dict) or "observation" not in payload:
                raise ValueError("Missing observation object")
            if set(payload) - {"observation", "expected_server_instance_id"}:
                raise ValueError("Unexpected request fields")
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            raise HTTPException(status_code=400, detail=f"Invalid request: {exc}") from exc
        if (
            "expected_server_instance_id" in payload
            and payload["expected_server_instance_id"] != metadata["server_instance_id"]
        ):
            raise HTTPException(
                status_code=409,
                detail="Inference server changed; review metadata before enabling control",
            )
        try:
            observation = validate_observation(payload["observation"])
            if (
                metadata.get("data_config") == "kmu_auv"
                and not (observation["state.validity"] == 1).all()
            ):
                raise ValueError("kmu_auv requires valid cameras, DVL velocity and altitude")
        except (ValueError, TypeError, OverflowError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        try:
            encoded_action = await run_in_threadpool(infer, observation)
            return Response(content=encoded_action, media_type="application/json")
        except Exception as exc:
            logging.exception("VLA inference failed")
            raise HTTPException(
                status_code=500, detail="Model inference failed; inspect the server log"
            ) from exc

    return app


def _load_policy(checkpoint: Path, data_config: str, device: str, denoising_steps: int):
    # Choose the bundled, reviewed runtime before importing any gr00t module.
    vendor = Path(__file__).resolve().parent / "_vendor"
    loaded = sys.modules.get("gr00t")
    if loaded is not None:
        loaded_path = getattr(loaded, "__file__", None)
        if loaded_path is None or not Path(loaded_path).resolve().is_relative_to(vendor):
            raise RuntimeError(
                "Another gr00t package is already loaded; start a fresh server process"
            )
    if not (vendor / "gr00t/model/policy.py").is_file():
        raise RuntimeError(
            "Bundled gr00t runtime is missing; run from the complete source checkout"
        )
    sys.path.insert(0, str(vendor))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    from gr00t.experiment.data_config import load_data_config
    from gr00t.model.policy import Gr00tPolicy

    config = load_data_config(data_config)
    return Gr00tPolicy(
        model_path=str(checkpoint),
        embodiment_tag="new_embodiment",
        modality_config=config.modality_config(),
        modality_transform=config.transform(),
        denoising_steps=denoising_steps,
        device=device,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data_config", choices=DATA_CONFIGS, default=DATA_CONFIGS[0])
    parser.add_argument(
        "--runtime_config", type=Path, help="Reviewed real settings; mandatory for legacy models"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--denoising_steps", type=int, default=4)
    parser.add_argument("--check_only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or args.denoising_steps <= 0:
        parser.error("port must be 1..65535 and denoising_steps must be positive")
    try:
        metadata = inspect_checkpoint(args.checkpoint, args.data_config, args.runtime_config)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.check_only:
        print(json.dumps(metadata, indent=2))
        return
    policy = _load_policy(
        args.checkpoint.expanduser().resolve(), args.data_config, args.device, args.denoising_steps
    )
    import uvicorn

    uvicorn.run(create_app(policy, metadata), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
