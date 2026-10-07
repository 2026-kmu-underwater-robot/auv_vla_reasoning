"""ROS sensor-to-HTTP-to-RC adapter; explicit enable plus live deadman required.

Run with ROS Humble Python and the vehicle's message packages sourced. This
module deliberately does not import torch, GR00T models or GPU dependencies.
"""

import signal
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import rclpy
from mavros_msgs.msg import OverrideRCIn, State
from rcl_interfaces.msg import SetParametersResult
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import Bool
from std_srvs.srv import SetBool

from .controls import CommandLimiter, motion_chunk, release_channels
from .observations import VlaObservation


def request_action(url, observation, timeout, server_instance_id=None):
    """Use the server's explicit encoded payload format, with bounded HTTP waits."""
    import json_numpy
    import requests

    payload = {"observation": observation}
    if server_instance_id is not None:
        payload["expected_server_instance_id"] = server_instance_id
    response = requests.post(
        url.rstrip("/") + "/act",
        json={"encoded": json_numpy.dumps(payload)},
        timeout=(timeout, timeout),
    )
    response.raise_for_status()
    return motion_chunk(json_numpy.loads(response.text))


def request_metadata(url, timeout):
    import requests

    response = requests.get(url.rstrip("/") + "/metadata", timeout=(timeout, timeout))
    response.raise_for_status()
    return response.json()


class RovPolicyAdapter(Node):
    def __init__(self, sensors):
        super().__init__("vla_policy")
        if self.get_parameter("use_sim_time").value:
            raise ValueError("auv_vla_reasoning supports real-vehicle wall time only")
        self.add_on_set_parameters_callback(self._validate_parameters)
        self.sensors = sensors
        self.url = str(self.declare_parameter("server_url", "http://127.0.0.1:8000").value)
        self.dry_run = bool(self.declare_parameter("dry_run", True).value)
        self.expected_mode = str(self.declare_parameter("expected_mode", "STABILIZE").value)
        self.timeout = float(self.declare_parameter("max_action_age_sec", 0.3).value)
        if not np.isfinite(self.timeout) or not 0.1 <= self.timeout <= 1.0:
            raise ValueError("max_action_age_sec must be in [0.1, 1.0]")
        self.limiter = CommandLimiter(
            limit=float(self.declare_parameter("command_limit", 0.3).value),
            slew_per_second=float(self.declare_parameter("command_slew_per_sec", 0.5).value),
            neutral=sensors._neutral_pwm,
            span=sensors._pwm_span,
        )
        if sensors._action_channel_indices != (4, 5, 2, 3):
            raise ValueError("This adapter requires ArduSub channels [5, 6, 3, 4]")
        self.output = self.create_publisher(
            OverrideRCIn, "/vla/proposed_rc" if self.dry_run else "/mavros/rc/override", 10
        )
        self.create_subscription(Bool, "/vla/deadman", self._deadman, 10)
        self.create_subscription(State, "/mavros/state", self._state, qos_profile_sensor_data)
        self.create_service(SetBool, "/vla/enable", self._enable)
        self.enabled = False
        self.owned = False
        self.deadman_at = -float("inf")
        self.state_at = -float("inf")
        self.vehicle_state = None
        self.epoch = 0
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.chunk = None
        self.requested_at = 0.0
        self.chunk_at = 0.0
        self.active_task = None
        self.server_instance_id = None
        self.last_tick = time.monotonic()
        self.last_ros_time = sensors._now()
        self.clock_advanced_at = self.last_tick
        self.timer = self.create_timer(
            0.1, self._tick, clock=Clock(clock_type=ClockType.STEADY_TIME)
        )
        self.get_logger().info(f"VLA adapter disabled, dry_run={self.dry_run}; no arm/mode changes")

    def _deadman(self, message):
        if message.data:
            self.deadman_at = time.monotonic()
        else:
            self.deadman_at = -float("inf")
            self._stop()

    def _validate_parameters(self, parameters):
        if any(parameter.name == "use_sim_time" and parameter.value for parameter in parameters):
            return SetParametersResult(successful=False, reason="Real inference requires wall time")
        return SetParametersResult(successful=True)

    def _state(self, message):
        self.vehicle_state = message
        self.state_at = time.monotonic()

    def _ready(self, now):
        if now - self.deadman_at > 0.3:
            return False
        if self.dry_run:
            return True
        if self._competing_rc_publisher():
            return False
        state = self.vehicle_state
        return bool(
            state
            and now - self.state_at <= 1.0
            and state.connected
            and state.armed
            and state.mode == self.expected_mode
        )

    def _competing_rc_publisher(self):
        """Do not compete with joy2mavros or another controller for RC ownership."""
        publishers = self.get_publishers_info_by_topic(self.output.topic_name)
        return len(publishers) != 1 or (publishers[0].node_name, publishers[0].node_namespace) != (
            self.get_name(),
            self.get_namespace(),
        )

    def _check_server_settings(self):
        metadata = request_metadata(self.url, self.timeout)
        expected = {
            "action_channels": [5, 6, 3, 4],
            "fps": 10,
            "neutral_pwm": self.sensors._neutral_pwm,
            "pwm_span": self.sensors._pwm_span,
            "expected_mode": self.expected_mode,
            "data_source": "real",
            "use_sim_time": False,
            "action_horizon": 16,
            "action_order": ["surge", "sway", "heave", "yaw"],
            "target_loss_weight": 0,
        }
        for name, value in expected.items():
            if metadata.get(name) != value:
                raise ValueError(f"Model/runtime setting mismatch: {name}")
        if (
            not isinstance(metadata.get("server_instance_id"), str)
            or not metadata["server_instance_id"]
        ):
            raise ValueError("Server does not identify its running model instance")
        self.server_instance_id = metadata["server_instance_id"]

    def _enable(self, request, response):
        self._stop()
        if not request.data:
            response.success = True
            response.message = "VLA disabled and owned channels released"
            return response
        try:
            if not self._ready(time.monotonic()):
                raise ValueError(
                    "Live deadman and (for RC output) armed, connected expected mode required"
                )
            self._check_server_settings()
            if not self._ready(time.monotonic()):
                raise ValueError("Vehicle readiness or deadman expired while checking the server")
            observation = self.sensors.policy_observation(np.zeros(4))
            self.active_task = observation["annotation.human.action.task_description"][0]
        except Exception as error:
            response.success = False
            response.message = str(error)
            return response
        self.last_tick = time.monotonic()
        self.last_ros_time = self.sensors._now()
        self.clock_advanced_at = self.last_tick
        self.enabled = True
        response.success = True
        response.message = "Enabled; keep /vla/deadman true at 10 Hz"
        return response

    def _publish(self, channels):
        message = OverrideRCIn()
        message.channels = channels
        self.output.publish(message)

    def _stop(self):
        self.enabled = False
        self.epoch += 1
        self.chunk = None
        self.active_task = None
        self.limiter.reset()
        if self.owned:
            self.owned = False
            if self.context.ok():
                self._publish(release_channels())

    def _tick(self):
        now = time.monotonic()
        dt = now - self.last_tick
        self.last_tick = now
        ros_time = self.sensors._now()
        if ros_time > self.last_ros_time:
            self.clock_advanced_at = now
        reset = ros_time < self.last_ros_time
        self.last_ros_time = ros_time
        if not self.enabled:
            return
        if (
            reset
            or now - self.clock_advanced_at > self.timeout
            or not self._ready(now)
            or not 0 < dt <= 0.2
        ):
            self._stop()
            return
        try:
            observation = self.sensors.policy_observation(self.limiter.previous)
            task = observation["annotation.human.action.task_description"][0]
            if task != self.active_task:
                raise ValueError("Task changed; enable VLA again for the new instruction")
            if self.future is not None and self.future.done():
                future, self.future = self.future, None
                if self.request_epoch == self.epoch and now - self.requested_at <= self.timeout:
                    self.chunk = future.result()
                    self.chunk_at = self.requested_at
            if self.chunk is not None:
                age = now - self.chunk_at
                if age > self.timeout:
                    self._stop()
                    return
                # Commands are predicted at 10 Hz; discard elapsed steps after inference latency.
                action = self.chunk[min(int(age * 10), 15)]
                self._publish(self.limiter.apply(action, dt))
                self.owned = True
            if self.future is None:
                self.requested_at = now
                self.request_epoch = self.epoch
                self.future = self.pool.submit(
                    request_action, self.url, observation, self.timeout, self.server_instance_id
                )
            elif now - self.requested_at > self.timeout:
                self._stop()
        except Exception as error:
            self.get_logger().warning(f"VLA stopped: {error}")
            self._stop()

    def close(self):
        try:
            self._stop()
        finally:
            self.pool.shutdown(wait=False, cancel_futures=True)


def main():
    # Let KeyboardInterrupt run cleanup while the ROS context can still release RC.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    previous_term_handler = signal.getsignal(signal.SIGTERM)

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    sensors = adapter = None
    executor = SingleThreadedExecutor()
    try:
        sensors = VlaObservation()
        adapter = RovPolicyAdapter(sensors)
        executor.add_node(sensors)
        executor.add_node(adapter)
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            if adapter is not None:
                adapter.close()
        finally:
            executor.shutdown()
            if adapter is not None:
                adapter.destroy_node()
            if sensors is not None:
                sensors.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
            signal.signal(signal.SIGTERM, previous_term_handler)


if __name__ == "__main__":
    main()
