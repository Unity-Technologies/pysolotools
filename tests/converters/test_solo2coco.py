import os
import tempfile
from pathlib import Path

import pytest

from pysolotools.consumers.solo import Solo
from pysolotools.converters import SOLO2COCOConverter

pozole_sample_data = os.path.join(
    Path(__file__).parents[1], "data", "format_output_by_pozole", "attempt0"
)

mysterious_sample_data = os.path.join(
    Path(__file__).parents[1], "data", "mysterious_format_no_idea_the_source", "solo"
)


@pytest.mark.parametrize(
    "input_data_path, metadata_file_path, annotation_definitions_file_path",
    [
        (
            pozole_sample_data,
            os.path.join(pozole_sample_data, "metadata", "metadata.json"),
            os.path.join(pozole_sample_data, "metadata", "annotation_definitions.json"),
        ),
        (mysterious_sample_data, None, None),
    ],
)
def test_solo2coco_create_files(
    input_data_path, metadata_file_path, annotation_definitions_file_path
):
    converter = SOLO2COCOConverter()
    solo = Solo(
        data_path=input_data_path,
        metadata_file=metadata_file_path,
        annotation_definitions_file=annotation_definitions_file_path,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        converter.convert(solo, {"output_path": tmp_dir})
        expected_file1 = Path(tmp_dir) / "coco" / "semantic.json"
        expected_file2 = Path(tmp_dir) / "coco" / "instances.json"
        expected_image_folder = Path(tmp_dir) / "coco" / "images"

        assert expected_file1.exists()
        assert expected_file2.exists()
        assert len(list(expected_image_folder.glob("*"))) == 2
