import os
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from pysolotools.converters import Solo2YoloConverter


@patch("pysolotools.converters.solo2yolo.shutil")
def test_process_rgb_image(mock_shutil):
    mock_rgb_capture = MagicMock()
    mock_rgb_capture.dimension = (100, 100)
    mock_output = Path("test")
    image_id = 0
    data_root = Path("test")
    sequence_num = "test"
    Solo2YoloConverter._process_rgb_image(
        image_id, mock_rgb_capture, mock_output, data_root, sequence_num
    )

    mock_shutil.copy.assert_called_once()


@patch("pysolotools.converters.solo2yolo.Solo2YoloConverter._filter_annotation")
def test_process_annotations(mock_filter_annotation):
    image_id = 0
    mock_rgb_capture = MagicMock()
    mock_rgb_capture.dimension = (100, 100)
    mock_output = Path("test")
    mock_filename = "test" + os.path.sep + "camera_0.txt"

    with patch("pysolotools.converters.solo2yolo.open", mock_open()) as mocked_file:
        Solo2YoloConverter._process_annotations(image_id, mock_rgb_capture, mock_output)

        assert mock_filter_annotation.call_count == 1

        mocked_file.assert_called_once_with(mock_filename, "w")


def test_to_yolo_bbox():
    x, y, w, h = Solo2YoloConverter._to_yolo_bbox(100, 100, 0, 0, 100, 100)
    assert x == pytest.approx(0.5)
    assert y == pytest.approx(0.5)
    assert w == pytest.approx(1)
    assert h == pytest.approx(1)

    x, y, w, h = Solo2YoloConverter._to_yolo_bbox(100, 100, 0, 0, 50, 50)
    assert x == pytest.approx(0.25)
    assert y == pytest.approx(0.25)
    assert w == pytest.approx(0.5)
    assert h == pytest.approx(0.5)

    x, y, w, h = Solo2YoloConverter._to_yolo_bbox(100, 100, 50, 50, 50, 50)
    assert x == pytest.approx(0.75)
    assert y == pytest.approx(0.75)
    assert w == pytest.approx(0.5)
    assert h == pytest.approx(0.5)

    x, y, w, h = Solo2YoloConverter._to_yolo_bbox(100, 100, 40, 40, 20, 20)
    assert x == pytest.approx(0.5)
    assert y == pytest.approx(0.5)
    assert w == pytest.approx(0.2)
    assert h == pytest.approx(0.2)
