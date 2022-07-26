import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from pysolotools.core import (
    BoundingBox2DAnnotation,
    BoundingBox3DAnnotation,
    Frame,
    InstanceSegmentationAnnotation,
    KeypointAnnotation,
    RGBCameraCapture,
    SemanticSegmentationAnnotation,
)
from pysolotools.core.iterators import FramesIterator
from pysolotools.core.models import DatasetAnnotations, DatasetMetadata


class TestFramesIterator(unittest.TestCase):
    mocked_d_metadata = MagicMock(DatasetMetadata)
    mocked_anno_def = MagicMock(DatasetAnnotations)
    data_path = os.path.join(Path(__file__).parents[2], "data", "solo")
    f_iter = FramesIterator(data_path, mocked_d_metadata(), mocked_anno_def())
    test_frame_path = f"{data_path}/sequence.0/step0.frame_data.json"
    test_frame = f_iter.parse_frame(test_frame_path)

    def test_iter_type(self):
        self.assertIsInstance(self.f_iter, FramesIterator)

    def test_parse_frame(self):

        self.assertIsInstance(self.test_frame, Frame)
        rgb_captures = filter(
            lambda k: isinstance(k, RGBCameraCapture), self.test_frame.captures
        )
        print(f"\n \n Testing with {len(list(rgb_captures))} RGBCameraCapture \n \n ")
        for capture in rgb_captures:
            self.assertIsInstance(capture, RGBCameraCapture)

    def test_annotations(self):
        rgb_captures = filter(
            lambda k: isinstance(k, RGBCameraCapture), self.test_frame.captures
        )
        for rgb_capture in rgb_captures:
            annotations = rgb_capture.annotations

            bbox2dannotations = filter(
                lambda c: isinstance(c, BoundingBox2DAnnotation), annotations
            )
            print(
                f"\n \n Testing with {len(list(bbox2dannotations))} BoundingBox2DAnnotation \n \n "
            )
            for bbox2d in bbox2dannotations:
                self.assertIsInstance(bbox2d, BoundingBox2DAnnotation)

            bbox3dannotations = filter(
                lambda c: isinstance(c, BoundingBox3DAnnotation), annotations
            )
            print(
                f"\n \n Testing with {len(list(bbox3dannotations))} BoundingBox3DAnnotation \n \n "
            )
            for bbox3d in bbox3dannotations:
                self.assertIsInstance(bbox3d, BoundingBox3DAnnotation)

            instance_segmentation_annotations = filter(
                lambda c: isinstance(c, InstanceSegmentationAnnotation), annotations
            )
            print(
                f"\n \n Testing with {len(list(instance_segmentation_annotations))} "
                f"InstanceSegmentationAnnotation \n \n "
            )
            for instance_segmentation_annotation in instance_segmentation_annotations:
                self.assertIsInstance(
                    instance_segmentation_annotation, InstanceSegmentationAnnotation
                )

            semantic_segmentation_annotations = filter(
                lambda c: isinstance(c, SemanticSegmentationAnnotation), annotations
            )
            print(
                f"\n \n Testing with {len(list(semantic_segmentation_annotations))} "
                f"SemanticSegmentationAnnotation \n \n "
            )
            for semantic_segmentation_annotation in semantic_segmentation_annotations:
                self.assertIsInstance(
                    semantic_segmentation_annotation, SemanticSegmentationAnnotation
                )

            keypoint_annotations = filter(
                lambda c: isinstance(c, KeypointAnnotation), annotations
            )
            print(
                f"\n \n Testing with {len(list(keypoint_annotations))} "
                f"SemanticSegmentationAnnotation \n \n "
            )
            for keypoint_annotation in keypoint_annotations:
                self.assertIsInstance(keypoint_annotation, KeypointAnnotation)
