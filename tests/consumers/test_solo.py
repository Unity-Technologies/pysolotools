import os
import unittest
from pathlib import Path

from pysolotools.consumers import Solo
from pysolotools.core.iterators import FramesIterator
from pysolotools.core.models import (
    AnnotationDefinition,
    DatasetMetadata,
    DefinitionFactory,
)


class TestSolo(unittest.TestCase):
    solo = Solo(os.path.join(Path(__file__).parents[1], "data", "solo"))

    def test_get_metadata(self):
        metadata = self.solo.get_metadata()
        self.assertIsInstance(metadata, DatasetMetadata)

    def test_get_annotation_definitions(self):
        annotation_def_types = list(DefinitionFactory.switcher.values())
        annotation_def_types.append(AnnotationDefinition)
        annotation_definition = self.solo.get_annotation_definitions()
        for ann_def in annotation_definition.annotationDefinitions:
            assert isinstance(ann_def, tuple(annotation_def_types))

    def test_frames(self):
        frames_iter = self.solo.frames()
        self.assertIsInstance(frames_iter, FramesIterator)

        with self.assertRaises(StopIteration):
            while True:
                next(frames_iter)

    def test_len(self):
        self.assertEqual(len(self.solo.frames()), 2)
