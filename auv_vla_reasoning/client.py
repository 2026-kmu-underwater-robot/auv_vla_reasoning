"""Predict actions from an NPZ observation without publishing robot commands."""

import argparse
from pathlib import Path

import json_numpy
import numpy as np
import requests

from .server import TASK_KEY, validate_action, validate_observation


def predict(url: str, observation: dict, timeout: float = 30.0) -> object:
    """Send one observation and return validated CAP-free model actions."""
    observation = validate_observation(observation)
    response = requests.post(
        url.rstrip("/") + "/act",
        data=json_numpy.dumps({"observation": observation}),
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )
    response.raise_for_status()
    return validate_action(json_numpy.loads(response.content))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--observation",
        required=True,
        type=Path,
        help="NPZ with camera and state keys; allow_pickle=False",
    )
    parser.add_argument("--task", help="Override the task description stored in the NPZ")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--output", type=Path, help="Optional JSON-numpy prediction output")
    args = parser.parse_args()
    if not np.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be finite and positive")
    with np.load(args.observation, allow_pickle=False) as archive:
        observation = dict(archive)
    if args.task is not None:
        observation[TASK_KEY] = [args.task]
    encoded = json_numpy.dumps(predict(args.url, observation, args.timeout))
    if args.output is not None:
        args.output.write_text(encoded + "\n")
    else:
        print(encoded)


if __name__ == "__main__":
    main()
