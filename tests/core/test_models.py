import json
import tempfile

import pytest

from pysolo.core.exceptions import MissingCaptureException
from pysolo.core.models import Frame, RGBCameraCapture


def test_frame_get_file_path():
    expected_path = "sequence.0/step0.camera.png"
    f_path = "tests/data/solo/sequence.0/step0.frame_data.json"
    with open(f_path, "r") as f:
        frame = Frame.from_json(f.read())
        rgb_img_path = frame.get_file_path(capture=RGBCameraCapture)
        assert rgb_img_path == expected_path


def test_frame_get_file_path_raises_exception():
    f_path = "tests/data/solo/sequence.0/step0.frame_data.json"
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


def test_frame_without_metadata():
    f_path = "tests/data/solo/sequence.0/step0.frame_data.json"
    with open(f_path, "r") as f:
        frame = Frame.from_json(f.read())
        assert frame.get_metadata() == {}


def test_frame_with_metadata():
    f_path = "tests/data/solo/sequence.1/step0.frame_data.json"
    test_metadata = {"custom-field-1": 10}
    with open(f_path, "r") as f:
        frame = Frame.from_json(f.read())
        assert frame.get_metadata() == test_metadata
