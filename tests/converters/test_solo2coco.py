import tempfile
from pathlib import Path

from pysolotools.converters import SOLO2COCOConverter


def test_solo2coco_create_files(solo_instance):
    converter = SOLO2COCOConverter(solo_instance)

    with tempfile.TemporaryDirectory() as tmp_dir:
        converter.convert(output_path=tmp_dir)
        expected_file1 = Path(tmp_dir) / "coco" / "semantic.json"
        expected_file2 = Path(tmp_dir) / "coco" / "instances.json"
        expected_image_folder = Path(tmp_dir) / "coco" / "images"

        assert expected_file1.exists()
        assert expected_file2.exists()
        assert len(list(expected_image_folder.glob("*"))) == 2
