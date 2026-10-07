"""ROS sensor observations for physical-robot VLA inference.

Extracted from ``VlaDataCollector.policy_observation`` and its sensor callbacks.
This node has no recording services, recording timer, dataset files, or RC
subscriptions, so it can coexist with the collector and its GUI.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from auv_dvl_a50_msg.msg import DVL
from geometry_msgs.msg import PoseWithCovarianceStamped, TwistWithCovarianceStamped
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image, Imu
from std_msgs.msg import String

from .sensor_math import (
    STATE_SLICES,
    body_velocity,
    build_state,
    normalize_quaternion_wxyz,
    sample_is_fresh,
)


@dataclass
class Latest:
    value: Any
    source_time: float
    received_time: float
    frame_id: str


class VlaObservation(Node):
    """Keep the latest real sensor readings and build a model observation."""

    def __init__(self, **node_kwargs: Any) -> None:
        super().__init__("vla_observation", **node_kwargs)
        if self.get_parameter("use_sim_time").value:
            raise ValueError("Physical inference requires use_sim_time=false")
        self.add_on_set_parameters_callback(self._validate_parameters)
        self._lock = threading.RLock()
        self._max_sensor_age = float(self.declare_parameter("max_sensor_age_sec", 0.25).value)
        if not np.isfinite(self._max_sensor_age) or self._max_sensor_age <= 0:
            raise ValueError("max_sensor_age_sec must be finite and positive")
        self._neutral_pwm = int(self.declare_parameter("neutral_pwm", 1500).value)
        self._pwm_span = int(self.declare_parameter("pwm_span", 300).value)
        if (
            self._pwm_span <= 0
            or self._neutral_pwm - self._pwm_span < 1000
            or self._neutral_pwm + self._pwm_span > 2000
        ):
            raise ValueError("neutral_pwm +/- pwm_span must fit within [1000, 2000]")
        channels = self.declare_parameter("action_channels", [5, 6, 3, 4]).value
        if list(channels) != [5, 6, 3, 4]:
            raise ValueError("ArduSub action_channels must be [5, 6, 3, 4]")
        self._action_channel_indices = (4, 5, 2, 3)
        self._depth_positive_down = bool(
            self.declare_parameter("depth_pose_z_is_positive_up", True).value
        )
        self._body_frame = str(self.declare_parameter("body_frame", "base_link").value)
        self._dvl_input_frame = str(self.declare_parameter("dvl_input_frame", "dvl_link").value)
        self._dvl_convention = str(self.declare_parameter("dvl_convention", "FRD").value)
        if not self._body_frame or not self._dvl_input_frame:
            raise ValueError("body_frame and dvl_input_frame must be nonempty")
        if self._dvl_convention not in ("FRD", "FLU"):
            raise ValueError("dvl_convention must be FRD or FLU")
        self._task_description = str(self.declare_parameter("default_task", "").value).strip()
        self._ego: Latest | None = None
        self._release: Latest | None = None
        self._dvl_twist: Latest | None = None
        self._dvl_data: Latest | None = None
        self._imu: Latest | None = None
        self._depth: Latest | None = None

        self._subscribe_camera("ego", "/imx219/camera0/image_raw/compressed", self._on_ego_image)
        self._subscribe_camera(
            "buoy_release", "/imx219/camera1/image_raw/compressed", self._on_release_image
        )
        for message_type, parameter, default, callback in (
            (TwistWithCovarianceStamped, "dvl_twist_topic", "/dvl/twist", self._on_dvl_twist),
            (DVL, "dvl_data_topic", "/dvl/data", self._on_dvl_data),
            (Imu, "imu_topic", "/mavros/imu/data", self._on_imu),
            (PoseWithCovarianceStamped, "depth_topic", "/depth/pose", self._on_depth),
        ):
            topic = str(self.declare_parameter(parameter, default).value)
            self.create_subscription(message_type, topic, callback, qos_profile_sensor_data)
        task_topic = str(
            self.declare_parameter("task_description_topic", "/vla/task_description").value
        )
        self.create_subscription(String, task_topic, self._on_task_description, 10)

    def _validate_parameters(self, parameters: list) -> SetParametersResult:
        if any(parameter.name == "use_sim_time" and parameter.value for parameter in parameters):
            return SetParametersResult(
                successful=False, reason="Physical inference requires use_sim_time=false"
            )
        return SetParametersResult(successful=True)

    def _subscribe_camera(self, name: str, default: str, callback: Any) -> None:
        topic = str(self.declare_parameter(f"{name}_image_topic", default).value)
        transport = str(self.declare_parameter(f"{name}_image_transport", "compressed").value)
        if transport not in ("compressed", "raw"):
            raise ValueError(f"{name}_image_transport must be compressed or raw")
        message_type = CompressedImage if transport == "compressed" else Image
        self.create_subscription(message_type, topic, callback, qos_profile_sensor_data)

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _latest(self, message: Any, value: Any) -> Latest:
        received = self._now()
        header = getattr(message, "header", None)
        stamp = getattr(header, "stamp", None)
        source = received if stamp is None else float(stamp.sec) + float(stamp.nanosec) * 1.0e-9
        return Latest(value, source, received, str(getattr(header, "frame_id", "")))

    @staticmethod
    def _decode_image(message: CompressedImage | Image) -> np.ndarray:
        if isinstance(message, CompressedImage):
            try:
                image = cv2.imdecode(np.frombuffer(message.data, dtype=np.uint8), cv2.IMREAD_COLOR)
            except cv2.error as error:
                raise ValueError("Cannot decode compressed camera image") from error
            if image is None:
                raise ValueError("Cannot decode compressed camera image")
            return image
        if message.encoding.lower() not in ("rgb8", "bgr8"):
            raise ValueError("Raw camera encoding must be rgb8 or bgr8")
        width, height, step = int(message.width), int(message.height), int(message.step)
        if width <= 0 or height <= 0 or step < width * 3 or len(message.data) != height * step:
            raise ValueError("Raw camera dimensions, stride, and payload disagree")
        image = np.frombuffer(message.data, dtype=np.uint8).reshape(height, step)
        image = image[:, : width * 3].reshape(height, width, 3).copy()
        if message.encoding.lower() == "rgb8":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    def _on_image(self, message: CompressedImage | Image, attribute: str) -> None:
        try:
            value = self._latest(message, self._decode_image(message))
        except ValueError as error:
            self.get_logger().warning(str(error))
            value = None
        with self._lock:
            setattr(self, attribute, value)

    def _on_ego_image(self, message: CompressedImage | Image) -> None:
        self._on_image(message, "_ego")

    def _on_release_image(self, message: CompressedImage | Image) -> None:
        self._on_image(message, "_release")

    def _on_dvl_twist(self, message: TwistWithCovarianceStamped) -> None:
        velocity = message.twist.twist.linear
        try:
            value = body_velocity(
                [velocity.x, velocity.y, velocity.z],
                message.header.frame_id,
                self._dvl_input_frame,
                self._dvl_convention,
            )
            latest = self._latest(message, value)
        except ValueError as error:
            self.get_logger().warning(str(error))
            latest = None
        with self._lock:
            self._dvl_twist = latest

    def _on_dvl_data(self, message: DVL) -> None:
        with self._lock:
            self._dvl_data = self._latest(
                message, (float(message.altitude), bool(message.velocity_valid))
            )

    def _on_imu(self, message: Imu) -> None:
        angular, linear, orientation = (
            message.angular_velocity,
            message.linear_acceleration,
            message.orientation,
        )
        try:
            if message.header.frame_id != self._body_frame:
                raise ValueError(
                    f"Expected IMU frame {self._body_frame!r}, got {message.header.frame_id!r}"
                )
            if any(
                covariance[0] < 0 or not np.all(np.isfinite(covariance))
                for covariance in (
                    message.orientation_covariance,
                    message.angular_velocity_covariance,
                    message.linear_acceleration_covariance,
                )
            ):
                raise ValueError(
                    "IMU reports unavailable orientation, angular velocity, or acceleration"
                )
            value = (
                np.asarray([angular.x, angular.y, angular.z], dtype=np.float32),
                np.asarray([linear.x, linear.y, linear.z], dtype=np.float32),
                normalize_quaternion_wxyz(
                    [orientation.w, orientation.x, orientation.y, orientation.z]
                ),
            )
            if not all(np.all(np.isfinite(part)) for part in value):
                raise ValueError("IMU contains a non-finite value")
            latest = self._latest(message, value)
        except ValueError as error:
            self.get_logger().warning(str(error))
            latest = None
        with self._lock:
            self._imu = latest

    def _on_depth(self, message: PoseWithCovarianceStamped) -> None:
        z_value = float(message.pose.pose.position.z)
        depth = -z_value if self._depth_positive_down else z_value
        with self._lock:
            self._depth = self._latest(message, depth) if np.isfinite(depth) else None

    def _on_task_description(self, message: String) -> None:
        with self._lock:
            self._task_description = message.data.strip()

    def _is_fresh(self, value: Latest | None, now: float) -> bool:
        return value is not None and sample_is_fresh(
            value.source_time, value.received_time, now, self._max_sensor_age
        )

    def policy_observation(self, previous_command: np.ndarray) -> dict:
        """Return two RGB views and 23 body-FLU state values for one timestep.

        Required cameras, IMU, and depth must be fresh. A nonempty instruction
        stays active until replaced; it is not a heartbeat. Missing or invalid
        DVL velocity/altitude are zeroed and identified by the validity flags.

        Raises:
            ValueError: Required inputs are missing, stale, or non-finite.
        """
        with self._lock:
            now = self._now()
            for name, value in (
                ("ego", self._ego),
                ("release", self._release),
                ("imu", self._imu),
                ("depth", self._depth),
            ):
                if not self._is_fresh(value, now):
                    raise ValueError(f"Missing/stale {name}")
            if not self._task_description:
                raise ValueError("Missing task description")
            raw_valid = self._is_fresh(self._dvl_data, now)
            altitude, velocity_valid = self._dvl_data.value if raw_valid else (0.0, False)
            dvl_valid = bool(raw_valid and velocity_valid and self._is_fresh(self._dvl_twist, now))
            altitude_valid = bool(raw_valid and np.isfinite(altitude) and altitude > 0)
            angular, linear, attitude = self._imu.value
            state = build_state(
                previous_command,
                self._dvl_twist.value if dvl_valid else np.zeros(3),
                angular,
                linear,
                attitude,
                self._depth.value,
                altitude if altitude_valid else 0.0,
                [1, 1, dvl_valid, altitude_valid],
            )
            observation = {
                f"state.{key}": state[start:end][None].copy()
                for key, (start, end) in STATE_SLICES.items()
            }
            observation.update(
                {
                    "video.ego": cv2.cvtColor(self._ego.value, cv2.COLOR_BGR2RGB)[None],
                    "video.buoy_release": cv2.cvtColor(self._release.value, cv2.COLOR_BGR2RGB)[
                        None
                    ],
                    "annotation.human.action.task_description": [self._task_description],
                }
            )
            return observation
