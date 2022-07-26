import os
import unittest
from pathlib import Path

from pysolotools.consumers import Solo
from pysolotools.core.iterators import FramesIterator
from pysolotools.core.models import DatasetMetadata, DefinitionFactory
from pysolotools.core.stats import SoloStats


class TestSolo(unittest.TestCase):
    solo = Solo(os.path.join(Path(__file__).parents[2], "data", "solo"))

    def test_get_metadata(self):
        metadata = self.solo.get_metadata()
        self.assertIsInstance(metadata, DatasetMetadata)

    def test_get_annotation_definitions(self):
        annotation_def_types = DefinitionFactory.switcher.values()
        annotation_definition = self.solo.get_annotation_definitions()
        for ann_def in annotation_definition.annotationDefinitions:
            self.assertIn(type(ann_def), annotation_def_types)

    def test_frames(self):
        frames_iter = self.solo.frames()
        self.assertIsInstance(frames_iter, FramesIterator)

        with self.assertRaises(StopIteration):
            while True:
                next(frames_iter)

    def test_stats(self):
        solo_stats = self.solo.stats
        self.assertIsInstance(solo_stats, SoloStats)

    def test_len(self):
        self.assertEqual(len(self.solo.frames()), 1)
