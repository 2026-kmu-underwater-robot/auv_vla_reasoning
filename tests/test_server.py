import concurrent.futures
import threading
import time
import types

import json_numpy
import numpy as np
import pytest
from fastapi.testclient import TestClient

from auv_vla_reasoning.controls import STATE_DIMS
from auv_vla_reasoning.server import TASK_KEY, _load_policy, create_app


class Policy:
    def __init__(self, result=None):
        self.result = (
            ({"action.motion": np.zeros((16, 4), dtype=np.float32)}, None)
            if result is None
            else result
        )
        self.calls = 0
        self.observation = None

    def get_action(self, observation):
        self.calls += 1
        self.observation = observation
        return self.result


@pytest.fixture
def observation():
    return {
        **{
            f"state.{name}": np.zeros((1, width), dtype=np.float32)
            for name, width in STATE_DIMS.items()
        },
        "video.ego": np.zeros((1, 4, 6, 3), dtype=np.uint8),
        "video.buoy_release": np.zeros((1, 4, 6, 3), dtype=np.uint8),
        TASK_KEY: ["Approach the red buoy."],
    }


def post(client, observation, **extra):
    return client.post(
        "/act",
        content=json_numpy.dumps({"observation": observation, **extra}),
        headers={"Content-Type": "application/json"},
    )


@pytest.mark.parametrize("encoded", [False, True])
def test_ndarray_round_trip_and_cap_free_tuple(observation, encoded):
    policy = Policy()
    client = TestClient(create_app(policy, {"pwm_span": 300}))
    payload = json_numpy.dumps({"observation": observation})
    response = (
        client.post("/act", json={"encoded": payload}) if encoded else post(client, observation)
    )
    assert response.status_code == 200, response.text
    result = json_numpy.loads(response.content)
    assert result[1] is None
    assert result[0]["action.motion"].shape == (16, 4)
    assert policy.observation["video.ego"].dtype == np.uint8
    assert policy.observation["state.depth"].dtype == np.float32


def test_plain_json_arrays_are_supported(observation):
    client = TestClient(create_app(Policy(), {}))
    plain = {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in observation.items()
    }
    assert client.post("/act", json={"observation": plain}).status_code == 200


def test_kmu_auv_sends_only_training_state_and_rejects_invalid_dvl(observation):
    policy = Policy()
    client = TestClient(create_app(policy, {"data_config": "kmu_auv"}))
    observation["state.validity"][:] = 1
    assert post(client, observation).status_code == 200
    assert "state.validity" not in policy.observation
    assert {key for key in policy.observation if key.startswith("state.")} == {
        "state.attitude",
        "state.dvl_velocity",
        "state.depth",
        "state.prev_command",
        "state.angular_velocity",
        "state.linear_acceleration",
        "state.altitude",
    }
    observation["state.validity"][0, 2] = 0
    assert post(client, observation).status_code == 422
    assert policy.calls == 1


@pytest.mark.parametrize(
    "kind",
    [
        "missing",
        "extra",
        "nan",
        "shape",
        "camera_dtype",
        "camera_shape",
        "task",
        "validity",
        "previous_command",
    ],
)
def test_bad_observation_does_not_invoke_policy(observation, kind):
    policy = Policy()
    client = TestClient(create_app(policy, {}))
    if kind == "missing":
        del observation["state.depth"]
    elif kind == "extra":
        observation["target_pos.target_pos"] = [[0] * 6]
    elif kind == "nan":
        observation["state.depth"][0, 0] = np.nan
    elif kind == "shape":
        observation["state.attitude"] = np.zeros((1, 3))
    elif kind == "camera_dtype":
        observation["video.ego"] = observation["video.ego"].astype(np.float32)
    elif kind == "camera_shape":
        observation["video.ego"] = np.zeros((4, 6, 3), dtype=np.uint8)
    elif kind == "task":
        observation[TASK_KEY] = [""]
    elif kind == "validity":
        observation["state.validity"][0, 0] = 0.5
    else:
        observation["state.prev_command"][0, 0] = 1.2
    assert post(client, observation).status_code == 422
    assert policy.calls == 0


@pytest.mark.parametrize(
    "body", [b"broken", b"{}", b"[]", b'{"encoded": 4}', b'{"encoded":"{}","other":1}']
)
def test_malformed_request_is_400(body):
    policy = Policy()
    response = TestClient(create_app(policy, {})).post("/act", content=body)
    assert response.status_code == 400
    assert policy.calls == 0


@pytest.mark.parametrize(
    "result",
    [
        ({"action.motion": np.zeros((16, 4))}, np.zeros(6)),
        {"action.motion": np.zeros((2, 16, 4))},
        {"action.motion": np.full((16, 4), np.nan)},
        {"action.motion": np.full((16, 4), 1.000001)},
        {"action.motion": np.full((16, 4), -128, dtype=np.int8)},
        {"action.motion": np.zeros((16, 4)), "target_pos": []},
    ],
)
def test_model_output_errors_are_500_and_never_clipped(observation, result):
    client = TestClient(create_app(Policy(result), {}))
    response = post(client, observation)
    assert response.status_code == 500
    assert "action.motion" not in response.json()


def test_server_restart_is_rejected_before_inference(observation):
    policy = Policy()
    client = TestClient(create_app(policy, {"pwm_span": 300, "fps": 10}))
    metadata = client.get("/metadata").json()
    assert metadata["pwm_span"] == 300
    assert client.get("/health").json()["server_instance_id"] == metadata["server_instance_id"]
    assert post(client, observation, expected_server_instance_id="old-instance").status_code == 409
    assert policy.calls == 0
    assert (
        post(
            client, observation, expected_server_instance_id=metadata["server_instance_id"]
        ).status_code
        == 200
    )


def test_backend_exception_is_server_error(observation):
    class BrokenPolicy:
        def get_action(self, observation):
            raise RuntimeError("private/checkpoint/path")

    response = post(TestClient(create_app(BrokenPolicy(), {})), observation)
    assert response.status_code == 500
    assert "private/checkpoint/path" not in response.text


def test_policy_calls_are_serialized(observation):
    class ConcurrentPolicy(Policy):
        def __init__(self):
            super().__init__()
            self.active = 0
            self.maximum_active = 0
            self.lock = threading.Lock()

        def get_action(self, observation):
            with self.lock:
                self.active += 1
                self.maximum_active = max(self.maximum_active, self.active)
            time.sleep(0.02)
            with self.lock:
                self.active -= 1
            return super().get_action(observation)

    policy = ConcurrentPolicy()
    client = TestClient(create_app(policy, {}))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        statuses = list(pool.map(lambda _: post(client, observation).status_code, range(4)))
    assert statuses == [200] * 4
    assert policy.maximum_active == 1


def test_runtime_rejects_preloaded_external_gr00t(monkeypatch, tmp_path):
    external = types.ModuleType("gr00t")
    external.__file__ = str(tmp_path / "gr00t/__init__.py")
    monkeypatch.setitem(__import__("sys").modules, "gr00t", external)
    with pytest.raises(RuntimeError, match="Another gr00t package"):
        _load_policy(tmp_path, "kmu26_auv_real", "cpu", 4)


def test_client_prediction_uses_numpy_transport_without_robot_output(observation, monkeypatch):
    from auv_vla_reasoning import client

    transport = TestClient(create_app(Policy(), {}))

    def request(url, *, data, headers, timeout):
        assert url == "http://testserver/act"
        assert timeout == 3.0
        return transport.post("/act", content=data, headers=headers)

    monkeypatch.setattr(client.requests, "post", request)
    result = client.predict("http://testserver", observation, 3.0)
    assert result[0]["action.motion"].shape == (16, 4)
