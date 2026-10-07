"""Launch only the inference ROS adapter; vehicle/cameras remain GUI-owned."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default = str(Path(get_package_share_directory("auv_vla_reasoning")) / "config/real.yaml")
    return LaunchDescription(
        [
            DeclareLaunchArgument("config", default_value=default),
            Node(
                package="auv_vla_reasoning",
                executable="auv-vla-ros",
                output="screen",
                parameters=[LaunchConfiguration("config")],
            ),
        ]
    )
