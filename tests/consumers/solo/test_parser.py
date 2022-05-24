import unittest

from pysolo.consumers import Solo
from pysolo.core.models import (
    DataFactory,
    DatasetMetadata,
    DefinitionFactory,
    Frame,
    RGBCameraCapture,
)


class TestSolo(unittest.TestCase):
    solo = Solo("tests/data/solo")
    f_path = "tests/data/solo/sequence.0/step0.frame_data.json"
    frame = solo.parse_frame(f_path)

    def test_get_metadata(self):
        metadata = self.solo.get_metadata()
        self.assertIsInstance(metadata, DatasetMetadata)

    def test_get_annotation_definitions(self):
        annotation_def_types = DefinitionFactory.switcher.values()
        for ann_def in self.solo.get_annotation_definitions():
            self.assertIn(type(ann_def), annotation_def_types)

    def test_parse_frame(self):
        self.assertIsInstance(self.frame, Frame)
        capture = self.frame.captures[0]
        self.assertIsInstance(capture, RGBCameraCapture)
        annotations = capture.annotations
        annotation_types = DataFactory.switcher.keys()
        for anno in annotations:
            self.assertIn(anno.type, annotation_types)

    def test_iter(self):
        """Should throw StopIteration on empty"""
        with self.assertRaises(StopIteration):
            while True:
                next(self.solo)

    def test_len(self):
        self.assertEqual(len(self.solo), 1)
