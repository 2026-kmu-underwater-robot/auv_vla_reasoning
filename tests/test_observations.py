"""Physical sensor semantics and recorder-independent ROS observations."""

import numpy as np
import pytest

from auv_vla_reasoning.sensor_math import body_velocity, build_state, sample_is_fresh


@pytest.mark.parametrize(
    "source,receipt,now,expected",
    [
        (10, 10, 10.1, True),
        (9, 10, 10.1, False),
        (10, 9, 10.1, False),
        (11, 10, 10.1, False),
        (10, 11, 10.1, False),
        (float("nan"), 10, 10.1, False),
    ],
)
def test_freshness_checks_capture_and_receipt(source, receipt, now, expected):
    assert sample_is_fresh(source, receipt, now, 0.25) == expected


def test_dvl_frd_converts_to_flu_without_changing_forward():
    np.testing.assert_array_equal(
        body_velocity([1, 2, 3], "dvl_link", "dvl_link", "FRD"), [1, -2, -3]
    )


@pytest.mark.parametrize("frame,values", [("wrong", [1, 2, 3]), ("dvl_link", [1, float("nan"), 3])])
def test_dvl_refuses_unknown_frame_and_nonfinite_velocity(frame, values):
    with pytest.raises(ValueError):
        body_velocity(values, frame, "dvl_link", "FRD")


def test_state_preserves_collector_layout_and_normalizes_wxyz():
    state = build_state(
        [0.1, 0.2, 0.3, 0.4], [1, 2, 3], [4, 5, 6], [7, 8, 9], [2, 0, 0, 0], 10, 11, [1, 1, 1, 1]
    )
    assert state.shape == (23,)
    assert state.dtype == np.float32
    np.testing.assert_allclose(
        state, [0.1, 0.2, 0.3, 0.4, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 0, 0, 0, 10, 11, 1, 1, 1, 1]
    )


@pytest.fixture
def node():
    rclpy = pytest.importorskip("rclpy")
    from auv_vla_reasoning.observations import VlaObservation

    rclpy.init(args=[])
    observation = VlaObservation()
    observation._now = lambda: 100.0
    yield observation
    observation.destroy_node()
    rclpy.shutdown()


def stamp(message, frame=""):
    message.header.stamp.sec = 100
    message.header.frame_id = frame
    return message


def seed(node):
    import cv2
    from auv_dvl_a50_msg.msg import DVL
    from geometry_msgs.msg import PoseWithCovarianceStamped, TwistWithCovarianceStamped
    from sensor_msgs.msg import CompressedImage, Imu
    from std_msgs.msg import String

    image = stamp(CompressedImage())
    image.data = cv2.imencode(".png", np.asarray([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8))[
        1
    ].tobytes()
    node._on_ego_image(image)
    node._on_release_image(image)
    imu = stamp(Imu(), "base_link")
    imu.orientation.w = 2.0
    imu.angular_velocity.x = 0.1
    imu.linear_acceleration.z = 9.8
    node._on_imu(imu)
    depth = stamp(PoseWithCovarianceStamped())
    depth.pose.pose.position.z = -2.5
    node._on_depth(depth)
    dvl = stamp(TwistWithCovarianceStamped(), "dvl_link")
    dvl.twist.twist.linear.x = 1.0
    dvl.twist.twist.linear.y = 2.0
    dvl.twist.twist.linear.z = 3.0
    node._on_dvl_twist(dvl)
    metadata = stamp(DVL())
    metadata.altitude = 4.0
    metadata.velocity_valid = True
    node._on_dvl_data(metadata)
    node._on_task_description(String(data="release the buoy"))
    return {"imu": imu, "depth": depth, "dvl": dvl, "metadata": metadata, "image": image}


def test_node_has_no_recording_timer_services_or_rc_subscription(node):
    assert node.get_name() == "vla_observation"
    assert list(node.timers) == []
    assert all("episode" not in service.srv_name for service in node.services)
    assert all("/rc/" not in subscription.topic_name for subscription in node.subscriptions)
    assert not hasattr(node, "_dataset_root")


def test_policy_observation_matches_training_views_state_and_units(node):
    seed(node)
    observation = node.policy_observation(np.asarray([0.1, 0.2, 0.3, 0.4]))
    assert set(observation) == {
        "state.prev_command",
        "state.dvl_velocity",
        "state.angular_velocity",
        "state.linear_acceleration",
        "state.attitude",
        "state.depth",
        "state.altitude",
        "state.validity",
        "video.ego",
        "video.buoy_release",
        "annotation.human.action.task_description",
    }
    assert (
        sum(value.shape[-1] for key, value in observation.items() if key.startswith("state.")) == 23
    )
    np.testing.assert_allclose(observation["state.prev_command"], [[0.1, 0.2, 0.3, 0.4]])
    np.testing.assert_array_equal(observation["state.dvl_velocity"], [[1, -2, -3]])
    np.testing.assert_array_equal(observation["state.attitude"], [[1, 0, 0, 0]])
    np.testing.assert_array_equal(observation["state.depth"], [[2.5]])
    np.testing.assert_array_equal(observation["state.altitude"], [[4]])
    np.testing.assert_array_equal(observation["state.validity"], [[1, 1, 1, 1]])
    np.testing.assert_array_equal(observation["video.ego"], [[[[3, 2, 1], [6, 5, 4]]]])


@pytest.mark.parametrize("attribute", ["_ego", "_release", "_imu", "_depth"])
def test_capture_staleness_stops_observation_even_if_just_received(node, attribute):
    seed(node)
    getattr(node, attribute).source_time = 99.0
    with pytest.raises(ValueError, match="Missing/stale"):
        node.policy_observation(np.zeros(4))


@pytest.mark.parametrize("attribute", ["_ego", "_release", "_imu", "_depth"])
def test_receipt_staleness_stops_observation(node, attribute):
    seed(node)
    getattr(node, attribute).received_time = 99.0
    with pytest.raises(ValueError, match="Missing/stale"):
        node.policy_observation(np.zeros(4))


def test_unavailable_dvl_uses_zero_values_and_validity(node):
    seed(node)
    node._dvl_data = None
    observation = node.policy_observation(np.zeros(4))
    np.testing.assert_array_equal(observation["state.dvl_velocity"], [[0, 0, 0]])
    np.testing.assert_array_equal(observation["state.altitude"], [[0]])
    np.testing.assert_array_equal(observation["state.validity"], [[1, 1, 0, 0]])


def test_invalid_dvl_velocity_can_preserve_valid_altitude(node):
    messages = seed(node)
    messages["metadata"].velocity_valid = False
    node._on_dvl_data(messages["metadata"])
    observation = node.policy_observation(np.zeros(4))
    np.testing.assert_array_equal(observation["state.dvl_velocity"], [[0, 0, 0]])
    np.testing.assert_array_equal(observation["state.validity"], [[1, 1, 0, 1]])


@pytest.mark.parametrize(
    "covariance",
    ["orientation_covariance", "angular_velocity_covariance", "linear_acceleration_covariance"],
)
def test_unavailable_imu_component_clears_cached_sample(node, covariance):
    imu = seed(node)["imu"]
    getattr(imu, covariance)[0] = -1.0
    node._on_imu(imu)
    with pytest.raises(ValueError, match="imu"):
        node.policy_observation(np.zeros(4))


def test_invalid_depth_clears_last_valid_value(node):
    depth = seed(node)["depth"]
    depth.pose.pose.position.z = float("nan")
    node._on_depth(depth)
    with pytest.raises(ValueError, match="depth"):
        node.policy_observation(np.zeros(4))


def test_empty_task_clears_previous_instruction(node):
    from std_msgs.msg import String

    seed(node)
    node._on_task_description(String(data="   "))
    with pytest.raises(ValueError, match="task description"):
        node.policy_observation(np.zeros(4))


def test_invalid_image_clears_last_valid_frame(node):
    image = seed(node)["image"]
    image.data = b""
    node._on_ego_image(image)
    with pytest.raises(ValueError, match="ego"):
        node.policy_observation(np.zeros(4))


@pytest.mark.parametrize(
    "encoding,data", [("rgb8", [3, 2, 1, 6, 5, 4, 0, 0]), ("bgr8", [1, 2, 3, 4, 5, 6, 0, 0])]
)
def test_explicit_raw_camera_transport_preserves_rgb_and_ignores_row_padding(node, encoding, data):
    from sensor_msgs.msg import Image

    seed(node)
    image = stamp(Image(height=1, width=2, encoding=encoding, step=8, data=bytes(data)))
    node._on_ego_image(image)
    observation = node.policy_observation(np.zeros(4))
    np.testing.assert_array_equal(observation["video.ego"], [[[[3, 2, 1], [6, 5, 4]]]])


def test_simulation_clock_cannot_be_enabled_after_start(node):
    from rclpy.parameter import Parameter

    result = node.set_parameters([Parameter("use_sim_time", value=True)])[0]
    assert not result.successful
    assert not node.get_parameter("use_sim_time").value


def test_simulation_clock_is_rejected_at_start():
    rclpy = pytest.importorskip("rclpy")
    from rclpy.parameter import Parameter

    from auv_vla_reasoning.observations import VlaObservation

    rclpy.init(args=[])
    try:
        with pytest.raises(ValueError, match="use_sim_time=false"):
            VlaObservation(parameter_overrides=[Parameter("use_sim_time", value=True)])
    finally:
        rclpy.shutdown()
