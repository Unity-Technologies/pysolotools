import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pysolotools.consumers import Solo
from pysolotools.converters import SOLO2COCOConverter


@pytest.fixture
def solo2coco_instance():
    mock_solo2coco = SOLO2COCOConverter()
    return mock_solo2coco


@pytest.fixture
def solo_instance():
    solo_path = "tests/mock_data/solo"
    mock_solo = Solo(solo_path)
    return mock_solo


def test_create_ann_file():
    file_name = "test.json"
    instances = ["test"]
    parent_dir = Path(__file__).parent.parent.absolute()
    with tempfile.TemporaryDirectory() as tmp_dir:
        output = parent_dir / Path(tmp_dir)
        SOLO2COCOConverter._create_ann_file(instances, output, file_name)
        expected_file = output / f"{file_name}"
        assert expected_file.exists()


@patch("pysolotools.converters.solo2coco.shutil")
def test_process_rgb_image(mock_shutil):
    mock_rgb_capture = MagicMock()
    mock_rgb_capture.dimension = (100, 100)
    mock_output = "test"
    image_id = "test"
    data_root = "test"
    sequence_num = "test"
    SOLO2COCOConverter._process_rgb_image(
        image_id, mock_rgb_capture, mock_output, data_root, sequence_num
    )

    mock_shutil.copy.assert_called_once()


@patch("pysolotools.converters.solo2coco.cv2")
def test_load_segmentation_image(mock_cv2):
    filename = "test"
    data_root = "test"
    sequence_num = "test"

    SOLO2COCOConverter._load_segmentation_image(filename, data_root, sequence_num)
    mock_cv2.imread.assert_called_once()
    mock_cv2.cvtColor.assert_called_once()


@patch("pysolotools.converters.solo2coco.mask_to_rle")
def test_compute_segmentation_map(mock_mask_to_rle):
    ins_image = np.zeros((3, 3, 3))
    color = np.zeros((3, 3, 3))
    SOLO2COCOConverter._compute_segmentation_map(ins_image=ins_image, color=color)
    mock_mask_to_rle.assert_called_once()


def test_keypoints_map():
    mock_solo_kp_map = MagicMock()
    mock_kp_ann = MagicMock()
    val = MagicMock()
    val.instanceId = 1
    mock_kp_ann.values = [val]
    return_val = SOLO2COCOConverter._keypoints_map(mock_solo_kp_map, mock_kp_ann)

    assert return_val == {1: (0, [])}


def test_sem_class_color_map():
    mock_sem_seg = MagicMock()
    val = MagicMock()
    val.labelName = "test_labelName"
    val.pixelValue = "test_pixelValue"
    mock_sem_seg.instances = [val]
    return_val = SOLO2COCOConverter._sem_class_color_map(mock_sem_seg)
    assert return_val == {"test_labelName": "test_pixelValue"}


@patch("pysolotools.converters.solo2coco.SOLO2COCOConverter._load_segmentation_image")
@patch("pysolotools.converters.solo2coco.SOLO2COCOConverter._sem_class_color_map")
@patch("pysolotools.converters.solo2coco.SOLO2COCOConverter._keypoints_map")
@patch("pysolotools.converters.solo2coco.SOLO2COCOConverter._filter_annotation")
def test_process_annotations(
    mock_filter_annotation,
    mok_keypoints_map,
    mock_sem_class_color_map,
    mock_load_segmentation_image,
):
    image_id = "test"
    rgb_capture = MagicMock()
    sequence_num = "test"
    data_root = "test"
    solo_kp_map = MagicMock()
    SOLO2COCOConverter._process_annotations(
        image_id, rgb_capture, sequence_num, data_root, solo_kp_map
    )
    assert mock_filter_annotation.call_count == 4  # filter all 4 types of annotations
    mok_keypoints_map.assert_called_once()
    mock_sem_class_color_map.assert_called_once()
    assert (
        mock_load_segmentation_image.call_count == 2
    )  # called in both instance and semantic seg
