import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pysolotools.consumers import Solo
from pysolotools.converters import SOLO2COCOConverter
from pysolotools.core.models import RGBCameraCapture

parent_dir = Path(__file__).parent.parent.absolute()
solo_sample_data = str(parent_dir / "data" / "solo")
pozole_sample_data = str(parent_dir / "data" / "format_output_by_pozole" / "attempt0")


@pytest.fixture
def solo2coco_instance():
    mock_solo2coco = SOLO2COCOConverter()
    return mock_solo2coco


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


@pytest.mark.parametrize(
    "input_data_path, metadata_file_path, annotation_definitions_file_path",
    [
        (
            pozole_sample_data,
            os.path.join(pozole_sample_data, "metadata", "metadata.json"),
            os.path.join(pozole_sample_data, "metadata", "annotation_definitions.json"),
        ),
        (solo_sample_data, None, None),
    ],
)
def test_categories(
    input_data_path,
    metadata_file_path,
    annotation_definitions_file_path,
    solo2coco_instance,
):

    solo = Solo(
        data_path=input_data_path,
        metadata_file=metadata_file_path,
        annotation_definitions_file=annotation_definitions_file_path,
    )
    solo2coco_instance._solo = solo
    return_value = solo2coco_instance._categories()

    expected = [
        {
            "id": 1,
            "name": "Crate",
            "supercategory": "default",
            "keypoints": [],
            "skeleton": [],
        },
        {
            "id": 2,
            "name": "Cube",
            "supercategory": "default",
            "keypoints": [],
            "skeleton": [],
        },
        {
            "id": 3,
            "name": "Box",
            "supercategory": "default",
            "keypoints": [],
            "skeleton": [],
        },
        {
            "id": 4,
            "name": "Terrain",
            "supercategory": "default",
            "keypoints": [],
            "skeleton": [],
        },
        {
            "id": 5,
            "name": "Character",
            "supercategory": "default",
            "keypoints": [],
            "skeleton": [],
        },
    ]

    assert return_value == expected


@patch("pysolotools.converters.solo2coco.SOLO2COCOConverter._categories")
@patch("pysolotools.converters.solo2coco.SOLO2COCOConverter._create_ann_file")
def test_write_out_annotations(
    mock_create_ann_file, mock_categories, solo2coco_instance
):
    output_dir = "test"
    solo2coco_instance._write_out_annotations(output_dir)
    mock_categories.assert_called_once()
    mock_create_ann_file.assert_called_once()


@patch("pysolotools.converters.solo2coco.SOLO2COCOConverter._process_annotations")
@patch("pysolotools.converters.solo2coco.SOLO2COCOConverter._process_rgb_image")
def test_process_instances(mock_process_rgb_image, mock_process_annotations):
    frame = MagicMock(spec=RGBCameraCapture)
    frame.sequence = []
    frame.captures = [MagicMock(spec=RGBCameraCapture)]
    idx = "test"
    output = "test"
    data_root = "test"
    solo_kp_map = MagicMock()

    mock_process_annotations.return_value = (None, None, None)
    SOLO2COCOConverter._process_instances(frame, idx, output, data_root, solo_kp_map)
    mock_process_rgb_image.assert_called_once()
    mock_process_annotations.assert_called_once()


@pytest.mark.parametrize(
    "input_data_path, metadata_file_path, annotation_definitions_file_path",
    [
        (
            pozole_sample_data,
            os.path.join(pozole_sample_data, "metadata", "metadata.json"),
            os.path.join(pozole_sample_data, "metadata", "annotation_definitions.json"),
        ),
        (solo_sample_data, None, None),
    ],
)
def test_get_solo_kp_map(
    input_data_path,
    metadata_file_path,
    annotation_definitions_file_path,
    solo2coco_instance,
):
    solo = Solo(
        data_path=input_data_path,
        metadata_file=metadata_file_path,
        annotation_definitions_file=annotation_definitions_file_path,
    )
    solo2coco_instance._solo = solo

    resturn_value = solo2coco_instance._get_solo_kp_map()

    expected = {
        0: "nose",
        1: "neck",
        2: "right_shoulder",
        3: "right_elbow",
        4: "right_wrist",
        5: "left_shoulder",
        6: "left_elbow",
        7: "left_wrist",
        8: "right_hip",
        9: "right_knee",
        10: "right_ankle",
        11: "left_hip",
        12: "left_knee",
        13: "left_ankle",
        14: "right_eye",
        15: "left_eye",
        16: "right_ear",
        17: "left_ear",
    }

    assert resturn_value == expected


def test_callback(solo2coco_instance):
    result = [["image"], ["box"], ["instance"], ["semantic"]]
    solo2coco_instance.callback(result)

    assert solo2coco_instance._bbox_annotations == ["box"]
    assert solo2coco_instance._instance_annotations == ["instance"]
    assert solo2coco_instance._semantic_annotations == ["semantic"]


@patch("pysolotools.converters.solo2coco.SOLO2COCOConverter._write_out_annotations")
@patch("pysolotools.converters.solo2coco.SOLO2COCOConverter._get_solo_kp_map")
def test_convert(mock_get_solo_kp_map, mock_write_out_annotations, solo2coco_instance):
    arguments = {"output_path": "test"}
    solo2coco_instance._pool = MagicMock()
    solo2coco_instance._pool = MagicMock()
    solo2coco_instance.convert(MagicMock(), arguments)
    mock_get_solo_kp_map.assert_called_once()
    mock_write_out_annotations.assert_called_once()
