"""Check the bundled inference transform without loading the GPU stack."""

import ast
from pathlib import Path

import pytest


def test_kmu_auv_preprocessing_matches_training_contract():
    source = (
        Path(__file__).resolve().parents[1]
        / "auv_vla_reasoning/_vendor/gr00t/experiment/data_config.py"
    )
    module = ast.parse(source.read_text())
    definition = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "KMUauvDataConfig"
    )
    factory_names = (
        "VideoToTensor",
        "VideoCrop",
        "VideoResize",
        "VideoColorJitter",
        "VideoToNumpy",
        "StateActionToTensor",
        "StateActionTransform",
        "ConcatTransform",
        "GR00TTransform",
    )
    namespace = {
        "BaseDataConfig": object,
        "ModalityTransform": object,
        "ComposedModalityTransform": lambda *, transforms: transforms,
    }
    for name in factory_names:
        namespace[name] = lambda _name=name, **kwargs: (_name, kwargs)
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(source), "exec"), namespace)
    config = namespace["KMUauvDataConfig"]()
    state_keys = [
        "state.attitude",
        "state.dvl_velocity",
        "state.depth",
        "state.prev_command",
        "state.angular_velocity",
        "state.linear_acceleration",
        "state.altitude",
    ]
    assert config.state_keys == state_keys
    assert config.video_keys == ["video.ego", "video.buoy_release"]
    assert config.action_keys == ["action.motion"]
    assert config.observation_indices == [0]
    assert config.action_indices == list(range(16))
    transforms = config.transform()
    assert [name for name, _ in transforms] == [
        "VideoToTensor",
        "VideoCrop",
        "VideoResize",
        "VideoColorJitter",
        "VideoToNumpy",
        "StateActionToTensor",
        "StateActionTransform",
        "StateActionToTensor",
        "StateActionTransform",
        "ConcatTransform",
        "GR00TTransform",
    ]
    assert transforms[1][1]["scale"] == 0.95
    assert transforms[2][1] == {
        "apply_to": config.video_keys,
        "height": 224,
        "width": 224,
        "interpolation": "linear",
    }
    assert transforms[6][1]["normalization_modes"] == {key: "min_max" for key in state_keys}
    assert transforms[8][1]["normalization_modes"] == {"action.motion": "min_max"}
    assert transforms[9][1]["state_concat_order"] == state_keys
    assert transforms[10][1] == {
        "state_horizon": 1,
        "action_horizon": 16,
        "max_state_dim": 64,
        "max_action_dim": 32,
    }


def test_training_min_max_normalizer_on_physical_state():
    torch = pytest.importorskip("torch")
    source = (
        Path(__file__).resolve().parents[1]
        / "auv_vla_reasoning/_vendor/gr00t/data/transform/state_action.py"
    )
    module = ast.parse(source.read_text())
    definition = next(
        node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "Normalizer"
    )
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(source), "exec"), namespace)
    normalizer = namespace["Normalizer"](
        mode="min_max", statistics={"min": [-2.0, 0.0, 3.0], "max": [2.0, 10.0, 3.0]}
    )
    normalized = normalizer.forward(torch.tensor([[1.0, 5.0, 3.0]]))
    torch.testing.assert_close(normalized, torch.tensor([[0.5, 0.0, 0.0]]))
