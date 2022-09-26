from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pysolotools.consumers import Solo
from pysolotools.core import KeypointAnnotation
from pysolotools.stats.analyzers.keypoint_analyzer import (
    AvgKPPerKPCat,
    KPPoseDict,
    _calc_dist,
    _calc_mid,
    _frame_keypoints,
    _is_torso_visible_or_labeled,
    _translate_and_scale_xy,
)

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
LABEL_INDEX_MAP = {
    "nose": 0,
    "neck": 1,
    "right_shoulder": 2,
    "right_elbow": 3,
    "right_wrist": 4,
    "left_shoulder": 5,
    "left_elbow": 6,
    "left_wrist": 7,
    "right_hip": 8,
    "right_knee": 9,
    "right_ankle": 10,
    "left_hip": 11,
    "left_knee": 12,
    "left_ankle": 13,
    "right_eye": 14,
    "left_eye": 15,
    "right_ear": 16,
    "left_ear": 17,
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


def test_frame_keypoints(solo_instance):
    frames = solo_instance.frames()
    actual = _frame_keypoints(next(frames))
    assert isinstance(actual[0], KeypointAnnotation)


def test_is_torso_visible_or_labeled(solo_instance):
    frames = solo_instance.frames()
    frame = next(frames)
    annotations = _frame_keypoints(frame)
    input = annotations[0].values[0].keypoints
    actual = _is_torso_visible_or_labeled(input, LABEL_INDEX_MAP)
    assert actual == 1


@patch("pysolotools.stats.analyzers.keypoint_analyzer._calc_dist")
def test_translate_and_scale_xy(mock_calc_dist, solo_instance):
    mock_calc_dist.return_value = 1
    frames = solo_instance.frames()
    frame = next(frames)
    annotations = _frame_keypoints(frame)
    input = annotations[0].values[0].keypoints
    x_loc, y_loc = [], []
    for kp in input:
        x_loc.append(kp.location[0])
        y_loc.append(kp.location[1])
    x_loc, y_loc = _translate_and_scale_xy(
        np.array(x_loc), np.array(y_loc), LABEL_INDEX_MAP
    )
    expected_x = [
        0.0,
        -10.043885500000044,
        -26.35545350000001,
        -19.71730050000002,
        -16.420285500000034,
        18.174453500000027,
        34.85694849999993,
        41.01496850000001,
        -14.369369500000005,
        -32.74510250000003,
        -45.20701650000001,
        14.369369500000005,
        21.416814499999987,
        32.94917250000003,
        -29.7548855,
        -16.00838550000003,
        0.0,
        0.0,
    ]
    expected_y = [
        0.0,
        -94.5238405,
        -79.83717949999999,
        -44.0830795,
        -10.338232500000004,
        -75.73247950000001,
        -29.527079500000013,
        12.056756500000006,
        -2.178320499999984,
        59.77306450000003,
        132.16912050000002,
        2.178320499999984,
        57.19725950000003,
        104.5034205,
        -124.36521049999999,
        -124.06702250000001,
        0.0,
        0.0,
    ]
    assert x_loc.tolist() == expected_x
    assert y_loc.tolist() == expected_y


def test_calc_dist():
    v1, v2 = (1, 1), (1, 1)
    actual = _calc_dist(v1, v2)
    assert actual == 0


def test_calc_mid():
    v1, v2 = (1, 1), (1, 1)
    actual = _calc_mid(v1, v2)
    assert actual == (1.0, 1.0)
