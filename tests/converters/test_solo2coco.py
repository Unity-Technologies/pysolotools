import os
import tempfile
from pathlib import Path

from pysolotools.consumers.solo import Solo
from pysolotools.converters import SOLO2COCOConverter


def test_solo2coco_create_files():
    input_data_path = os.path.join(Path(__file__).parents[1], "data", "solo")
    converter = SOLO2COCOConverter()
    solo = Solo(data_path=input_data_path)

    with tempfile.TemporaryDirectory() as tmp_dir:
        converter.convert(solo, {"output_path": tmp_dir})
        expected_file1 = Path(tmp_dir) / "coco" / "semantic.json"
        expected_file2 = Path(tmp_dir) / "coco" / "instances.json"
        expected_image_folder = Path(tmp_dir) / "coco" / "images"

        assert expected_file1.exists()
        assert expected_file2.exists()
        assert len(list(expected_image_folder.glob("*"))) == 2
