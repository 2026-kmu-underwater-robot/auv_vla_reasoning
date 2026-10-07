"""ROS ament data installation; Python metadata is defined in setup.cfg."""

from glob import glob
from pathlib import Path

from setuptools import setup

setup(
    long_description=(Path(__file__).resolve().parent / "README.md").read_text(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/auv_vla_reasoning"]),
        ("share/auv_vla_reasoning", ["package.xml"]),
        ("share/auv_vla_reasoning/config", glob("config/*")),
        ("share/auv_vla_reasoning/launch", glob("launch/*.launch.py")),
    ],
)
