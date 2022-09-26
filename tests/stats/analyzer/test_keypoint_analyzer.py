from unittest.mock import MagicMock, patch

import pytest

from pysolotools.consumers import Solo
from pysolotools.stats.analyzers.keypoint_analyzer import AvgKPPerKPCat, KPPoseDict

MOCK_DATA = {
    "nose": 1,
    "neck": 1,
    "right_shoulder": 1,
    "right_elbow": 1,
    "right_wrist": 1,
    "left_shoulder": 1,
    "left_elbow": 1,
    "left_wrist": 1,
    "right_hip": 1,
    "right_knee": 1,
    "right_ankle": 1,
    "left_hip": 1,
    "left_knee": 1,
    "left_ankle": 1,
    "right_eye": 1,
    "left_eye": 1,
    "right_ear": 1,
    "left_ear": 1,
}


@pytest.fixture
def solo_instance():
    input_data_path = "tests/data/solo"
    solo = Solo(data_path=input_data_path)
    return solo


@patch("pysolotools.stats.analyzers.keypoint_analyzer._translate_and_scale_xy")
@patch("pysolotools.stats.analyzers.keypoint_analyzer._is_torso_visible_or_labeled")
def test_kp_position_analyze(mock_torso, mock_scale, solo_instance):
    mock_torso.return_value = True
    mock_scale.return_value = [0], [0]
    frames = solo_instance.frames()
    kp_pose = KPPoseDict(anno_def=solo_instance.annotation_definitions)
    actual = kp_pose.analyze(next(frames))
    expected = {
        "nose": {"x": [], "y": []},
        "neck": {"x": [], "y": []},
        "right_shoulder": {"x": [], "y": []},
        "right_elbow": {"x": [], "y": []},
        "right_wrist": {"x": [], "y": []},
        "left_shoulder": {"x": [], "y": []},
        "left_elbow": {"x": [], "y": []},
        "left_wrist": {"x": [], "y": []},
        "right_hip": {"x": [], "y": []},
        "right_knee": {"x": [], "y": []},
        "right_ankle": {"x": [], "y": []},
        "left_hip": {"x": [], "y": []},
        "left_knee": {"x": [], "y": []},
        "left_ankle": {"x": [], "y": []},
        "right_eye": {"x": [], "y": []},
        "left_eye": {"x": [], "y": []},
        "right_ear": {"x": [], "y": []},
        "left_ear": {"x": [], "y": []},
    }
    assert actual == expected
    mock_torso.assert_called_once()
    mock_scale.assert_called_once()


@patch("pysolotools.stats.analyzers.keypoint_analyzer._reverse_map")
@patch("pysolotools.stats.analyzers.keypoint_analyzer._kp_label_dict")
def test_kp_position_merge(mock_kp_label_dict, mock_reverse_map):
    kp_pose = KPPoseDict(anno_def=MagicMock())
    results = {"nose": {"x": [0], "y": [0]}}
    result = {"nose": {"x": [0], "y": [0]}}
    actual = kp_pose.merge(results, result)
    expected = {"nose": {"x": [0, 0], "y": [0, 0]}}
    assert actual == expected
    mock_kp_label_dict.assert_called_once()
    mock_reverse_map.assert_called_once()


def test_avg_kp_analyze(solo_instance):
    frames = solo_instance.frames()
    kp_avg = AvgKPPerKPCat(anno_def=solo_instance.annotation_definitions)
    actual = kp_avg.analyze(next(frames))
    assert actual == MOCK_DATA


@patch("pysolotools.stats.analyzers.keypoint_analyzer._kp_label_dict")
def test_avg_kp_merge(mock_kp_label_dict):
    kp_avg = AvgKPPerKPCat(anno_def=MagicMock())
    kp_avg.kp_anno_count = 2
    actual = kp_avg.merge(MOCK_DATA, MOCK_DATA)
    assert actual == MOCK_DATA
    mock_kp_label_dict.assert_called_once()
