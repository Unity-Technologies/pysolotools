import json
import os
import tempfile
from dataclasses import fields
from pathlib import Path

import pytest

from pysolotools.consumers.solo import Solo
from pysolotools.core import DatasetMetadata
from pysolotools.core.exceptions import MissingCaptureException
from pysolotools.core.models import Frame, RGBCameraCapture


def test_frame_get_file_path():
    expected_path = "sequence.0/step0.camera.png"
    f_path = os.path.join(
        Path(__file__).parents[1], "data", "solo", "sequence.0", "step0.frame_data.json"
    )
    with open(f_path, "r") as f:
        frame = Frame.from_json(f.read())
        rgb_img_path = frame.get_file_path(capture=RGBCameraCapture)
        assert rgb_img_path == expected_path


def test_frame_with_unknown_annotation():
    f_path = os.path.join(
        Path(__file__).parents[1], "data", "solo", "sequence.1", "step0.frame_data.json"
    )
    with open(f_path, "r") as f:
        frame = Frame.from_json(f.read())
        rgb_captures = frame.filter_captures(RGBCameraCapture)
        annotations = rgb_captures[0].annotations
        # There are 3 annotations with 2 unknown annotation type.
        assert len(annotations) == 3


def test_frame_get_file_path_raises_exception():
    f_path = os.path.join(
        Path(__file__).parents[1], "data", "solo", "sequence.0", "step0.frame_data.json"
    )
    temp_f = tempfile.NamedTemporaryFile()

    with open(f_path, "r") as f:
        data = json.load(f)
        data["captures"] = []
        with open(temp_f.name, "w+") as tf:
            json.dump(data, tf)

    with pytest.raises(MissingCaptureException):
        frame = Frame.from_json(temp_f.read())
        frame.get_file_path(capture=RGBCameraCapture)

    temp_f.close()


def test_annotation_label_without_metadata():
    f_path = os.path.join(
        Path(__file__).parents[1], "data", "solo", "sequence.0", "step0.frame_data.json"
    )
    with open(f_path, "r") as f:
        frame = Frame.from_json(f.read())
        for capture in frame.captures:
            for annotation in capture.annotations:
                if "values" in [f.name for f in fields(annotation)]:
                    v = annotation.values
                else:
                    v = annotation.instances

                for annoLabel in v:
                    assert annoLabel.metadata == {}


@pytest.mark.parametrize(
    "test_input, expected",
    [
        (
            {
                "unityVersion": "1",
                "perceptionVersion": "2",
                "totalFrames": 3,
                "totalSequences": 4,
                "sensors": ["5"],
                "metricCollectors": ["6"],
                "annotators": ["8"],
            },
            DatasetMetadata(
                unityVersion="1",
                perceptionVersion="2",
                totalFrames=3,
                totalSequences=4,
                sensors=["5"],
                metricCollectors=["6"],
                scenarioActiveRandomizers=[],
                annotators=["8"],
            ),
        ),
        (
            {
                "unityVersion": "1",
                "perceptionVersion": "2",
                "totalFrames": 3,
                "totalSequences": 4,
                "sensors": ["5"],
                "metricCollectors": ["6"],
                "scenarioActiveRandomizers": ["7"],
                "annotators": ["8"],
                "simulationStartTime": "9",
                "simulationEndTime": "10",
                "renderPipeline": "11",
                "scenarioRandomSeed": 12.0,
            },
            DatasetMetadata(
                unityVersion="1",
                perceptionVersion="2",
                totalFrames=3,
                totalSequences=4,
                sensors=["5"],
                metricCollectors=["6"],
                scenarioActiveRandomizers=["7"],
                annotators=["8"],
                simulationStartTime="9",
                simulationEndTime="10",
                renderPipeline="11",
                scenarioRandomSeed=12.0,
            ),
        ),
    ],
)
def test_dataset_metadata_serialization(test_input, expected):
    actual = DatasetMetadata.from_json(json.dumps(test_input))
    assert actual == expected


def test_solo_read_unknown_ann_def():
    undefined_ann_def_id = "Extra Annotation"
    solo = Solo(data_path=os.path.join(Path(__file__).parents[1], "data", "solo"))
    annotation_definitions = solo.get_annotation_definitions().annotationDefinitions
    ann_def_ids = [ann_def.id for ann_def in annotation_definitions]
    assert undefined_ann_def_id in ann_def_ids


def test_solo_read_unknown_annotation():
    expected_annotation_types = [
        "type.unity.com/unity.solo.DepthAnnotation",
        "type.unity.com/unity.solo.BoundingBox2DAnnotation",
        "type.unity.com/unity.solo.ExtraAnnotation",
    ]
    f_path = os.path.join(
        Path(__file__).parents[1], "data", "solo", "sequence.1", "step0.frame_data.json"
    )
    with open(f_path, "r") as f:
        frame = Frame.from_json(f.read())
        for capture in frame.captures:
            assert expected_annotation_types == [
                annotation.type for annotation in capture.annotations
            ]
