from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("auv_dvl_a50_msg")
from std_msgs.msg import Bool
from std_srvs.srv import SetBool

from auv_vla_reasoning import ros as runtime


@pytest.fixture
def adapter(monkeypatch):
    rclpy.init()
    clock = [10.0]
    monkeypatch.setattr(runtime.time, "monotonic", lambda: clock[0])
    sensors = SimpleNamespace(
        _neutral_pwm=1500,
        _pwm_span=300,
        _action_channel_indices=(4, 5, 2, 3),
        _now=lambda: clock[0],
        policy_observation=lambda previous: {
            "previous": previous.copy(),
            "annotation.human.action.task_description": ["Approach the buoy."],
        },
    )
    node = runtime.RovPolicyAdapter(sensors)
    monkeypatch.setattr(node, "_check_server_settings", lambda: None)
    monkeypatch.setattr(node, "_competing_rc_publisher", lambda: False)
    node.active_task = "Approach the buoy."
    output = []
    node._publish = lambda channels: output.append(channels)
    monkeypatch.setattr(node.pool, "submit", lambda *args: Future())
    yield node, clock, output
    node.close()
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


def test_disabled_by_default_and_deadman_required(adapter):
    node, clock, output = adapter
    node._tick()
    assert not output
    response = node._enable(SetBool.Request(data=True), SetBool.Response())
    assert not response.success
    node._deadman(Bool(data=True))
    assert node._enable(SetBool.Request(data=True), SetBool.Response()).success
    clock[0] += 0.1
    node._tick()
    assert not output  # inference pending, no command yet


def test_completed_response_is_bounded_then_deadman_releases(adapter):
    node, clock, output = adapter
    node.enabled = True
    node.deadman_at = 10.0
    node.future = Future()
    node.future.set_result(np.ones((16, 4)))
    node.request_epoch = node.epoch
    node.requested_at = 10.0
    clock[0] = 10.1
    node._tick()
    assert len(output) == 1
    assert output[-1][4] == 1515
    node._deadman(Bool(data=False))
    assert not node.enabled
    assert output[-1][4] == 0
    assert output[-1][0] == 65535


def test_old_request_after_disable_cannot_reacquire_rc(adapter):
    node, clock, output = adapter
    node.future = Future()
    node.request_epoch = node.epoch
    node.requested_at = 10.0
    node._stop()
    node.enabled = True
    node.active_task = "Approach the buoy."
    node.deadman_at = 10.0
    node.future.set_result(np.ones((16, 4)))
    clock[0] = 10.1
    node._tick()
    assert not output


def test_clock_reset_and_inference_deadline_release(adapter):
    node, clock, output = adapter
    node.enabled = True
    node.owned = True
    node.deadman_at = 10.0
    node.sensors._now = lambda: 0.0
    clock[0] = 10.1
    node._tick()
    assert not node.enabled and output[-1][4] == 0


def test_live_output_requires_armed_connected_expected_mode(adapter):
    node, clock, _ = adapter
    node.dry_run = False
    node.deadman_at = 10
    for armed, mode, expected in [
        (False, "STABILIZE", False),
        (True, "ALT_HOLD", False),
        (True, "STABILIZE", True),
    ]:
        node._state(SimpleNamespace(connected=True, armed=armed, mode=mode))
        assert node._ready(10) is expected


def test_pending_inference_cannot_hold_rc_past_deadline(adapter):
    node, clock, output = adapter
    node.enabled = node.owned = True
    node.future = Future()
    node.requested_at = 9.6
    node.deadman_at = 10.0
    clock[0] = 10.1
    node._tick()
    assert not node.enabled and output[-1][4] == 0


def test_paused_ros_clock_releases_even_with_live_deadman(adapter):
    node, clock, output = adapter
    node.enabled = node.owned = True
    node.sensors._now = lambda: 10.0
    for now in (10.1, 10.2, 10.4):
        clock[0] = now
        node.deadman_at = now
        node._tick()
    assert not node.enabled and output[-1][4] == 0


def test_http_numpy_wire_format_round_trip():
    import json
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from threading import Thread

    import json_numpy

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            observation = json_numpy.loads(payload["encoded"])["observation"]
            assert observation["state.prev_command"].shape == (1, 4)
            data = json_numpy.dumps({"action.motion": np.zeros((1, 16, 4))}).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        response = runtime.request_action(
            f"http://127.0.0.1:{server.server_port}",
            {"state.prev_command": np.zeros((1, 4))},
            1.0,
        )
        assert response.shape == (16, 4)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_changed_task_discards_inflight_motion(adapter):
    node, clock, output = adapter
    node.enabled = node.owned = True
    node.deadman_at = 10.0
    node.sensors.policy_observation = lambda previous: {
        "annotation.human.action.task_description": ["A different task"]
    }
    clock[0] = 10.1
    node._tick()
    assert not node.enabled
    assert output[-1][4] == 0


def test_competing_controller_disables_live_motion(adapter, monkeypatch):
    node, clock, output = adapter
    node.enabled = node.owned = True
    node.dry_run = False
    node.deadman_at = 10.0
    node._state(SimpleNamespace(connected=True, armed=True, mode="STABILIZE"))
    monkeypatch.setattr(node, "_competing_rc_publisher", lambda: True)
    clock[0] = 10.1
    node._tick()
    assert not node.enabled
    assert output[-1][4] == 0


def test_metadata_mismatch_cannot_enable(adapter, monkeypatch):
    node, clock, _ = adapter
    node.deadman_at = 10.0
    monkeypatch.setattr(
        node,
        "_check_server_settings",
        lambda: runtime.RovPolicyAdapter._check_server_settings(node),
    )
    metadata = {
        "action_channels": [5, 6, 3, 4],
        "fps": 10,
        "neutral_pwm": 1500,
        "pwm_span": 400,
        "expected_mode": "STABILIZE",
        "data_source": "real",
        "use_sim_time": False,
        "action_horizon": 16,
        "action_order": ["surge", "sway", "heave", "yaw"],
        "target_loss_weight": 0,
        "server_instance_id": "test-model-instance",
    }
    monkeypatch.setattr(runtime, "request_metadata", lambda *args: metadata)
    response = node._enable(SetBool.Request(data=True), SetBool.Response())
    assert not response.success and "pwm_span" in response.message
    assert not node.enabled
    metadata["pwm_span"] = 300
    assert node._enable(SetBool.Request(data=True), SetBool.Response()).success
    assert node.server_instance_id == "test-model-instance"


def test_publisher_detection_includes_duplicate_names(adapter, monkeypatch):
    node, _, _ = adapter
    own = SimpleNamespace(node_name=node.get_name(), node_namespace=node.get_namespace())
    monkeypatch.setattr(node, "get_publishers_info_by_topic", lambda topic: [own])
    assert not runtime.RovPolicyAdapter._competing_rc_publisher(node)
    monkeypatch.setattr(node, "get_publishers_info_by_topic", lambda topic: [own, own])
    assert runtime.RovPolicyAdapter._competing_rc_publisher(node)


def test_obsolete_failed_request_cannot_stop_new_enable(adapter):
    node, clock, output = adapter
    node.future = Future()
    node.request_epoch = node.epoch
    node.requested_at = 10.0
    node._stop()
    node.enabled = True
    node.active_task = "Approach the buoy."
    node.deadman_at = 10.0
    node.future.set_exception(RuntimeError("Previous HTTP request timed out"))
    clock[0] = 10.1
    node._tick()
    assert node.enabled and not output


def test_close_after_external_ros_shutdown_finishes(adapter):
    node, _, _ = adapter
    node.enabled = node.owned = True
    rclpy.shutdown()
    node.close()
    assert not node.enabled and not node.owned


def test_close_releases_owned_channels_before_shutdown(adapter):
    node, _, output = adapter
    node.enabled = node.owned = True
    node.close()
    assert output[-1][4] == 0
    assert node.context.ok()


def test_metadata_latency_does_not_look_like_control_stall(adapter, monkeypatch):
    node, clock, _ = adapter
    node.deadman_at = clock[0]

    def check():
        clock[0] += 0.21

    monkeypatch.setattr(node, "_check_server_settings", check)
    assert node._enable(SetBool.Request(data=True), SetBool.Response()).success
    clock[0] += 0.05
    node._tick()
    assert node.enabled


def test_deadman_expiring_during_enable_cannot_enable(adapter, monkeypatch):
    node, clock, _ = adapter
    node.deadman_at = clock[0]

    def check():
        clock[0] += 0.31

    monkeypatch.setattr(node, "_check_server_settings", check)
    assert not node._enable(SetBool.Request(data=True), SetBool.Response()).success


def test_policy_rejects_simulation_clock_change(adapter):
    from rclpy.parameter import Parameter

    node, _, _ = adapter
    result = node.set_parameters([Parameter("use_sim_time", value=True)])
    assert not result[0].successful
    assert node.get_parameter("use_sim_time").value is False
