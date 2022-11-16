import logging
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from pysolotools.consumers import Solo
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

logger = logging.getLogger(__name__)


@dataclass
class FrameIteratorTest:
    solo: Solo
    frame: Frame
    frames_iterator: FramesIterator


@pytest.fixture
def setup_frame_iterator(solo_instance) -> FrameIteratorTest:
    mocked_d_metadata = MagicMock(DatasetMetadata)
    mocked_anno_def = MagicMock(DatasetAnnotations)
    frame_iterator = FramesIterator(
        solo_instance.data_path, mocked_d_metadata(), mocked_anno_def()
    )
    return FrameIteratorTest(
        solo=solo_instance,
        frames_iterator=frame_iterator,
        frame=frame_iterator.parse_frame(
            f"{solo_instance.data_path}/sequence.0/step0.frame_data.json"
        ),
    )


class TestFramesIterator:
    def test_iter_type(self, setup_frame_iterator: FrameIteratorTest):
        assert isinstance(setup_frame_iterator.frames_iterator, FramesIterator)

    def test_parse_frame(self, setup_frame_iterator: FrameIteratorTest):

        assert isinstance(setup_frame_iterator.frame, Frame)
        rgb_captures = filter(
            lambda k: isinstance(k, RGBCameraCapture),
            setup_frame_iterator.frame.captures,
        )
        logger.debug(
            f"\n \n Testing with {len(list(rgb_captures))} RGBCameraCapture \n \n "
        )
        for capture in rgb_captures:
            assert isinstance(capture, RGBCameraCapture)

    def test_annotations(self, setup_frame_iterator: FrameIteratorTest):
        rgb_captures = filter(
            lambda k: isinstance(k, RGBCameraCapture),
            setup_frame_iterator.frame.captures,
        )
        for rgb_capture in rgb_captures:
            annotations = rgb_capture.annotations

            bbox2dannotations = filter(
                lambda c: isinstance(c, BoundingBox2DAnnotation), annotations
            )
            logger.debug(
                f"\n \n Testing with {len(list(bbox2dannotations))} BoundingBox2DAnnotation \n \n "
            )
            for bbox2d in bbox2dannotations:
                assert isinstance(bbox2d, BoundingBox2DAnnotation)

            bbox3dannotations = filter(
                lambda c: isinstance(c, BoundingBox3DAnnotation), annotations
            )
            logger.debug(
                f"\n \n Testing with {len(list(bbox3dannotations))} BoundingBox3DAnnotation \n \n "
            )
            for bbox3d in bbox3dannotations:
                assert isinstance(bbox3d, BoundingBox3DAnnotation)

            instance_segmentation_annotations = filter(
                lambda c: isinstance(c, InstanceSegmentationAnnotation), annotations
            )
            logger.debug(
                f"\n \n Testing with {len(list(instance_segmentation_annotations))} "
                f"InstanceSegmentationAnnotation \n \n "
            )
            for instance_segmentation_annotation in instance_segmentation_annotations:
                assert isinstance(
                    instance_segmentation_annotation, InstanceSegmentationAnnotation
                )

            semantic_segmentation_annotations = filter(
                lambda c: isinstance(c, SemanticSegmentationAnnotation), annotations
            )
            logger.debug(
                f"\n \n Testing with {len(list(semantic_segmentation_annotations))} "
                f"SemanticSegmentationAnnotation \n \n "
            )
            for semantic_segmentation_annotation in semantic_segmentation_annotations:
                assert isinstance(
                    semantic_segmentation_annotation, SemanticSegmentationAnnotation
                )

            keypoint_annotations = filter(
                lambda c: isinstance(c, KeypointAnnotation), annotations
            )
            logger.debug(
                f"\n \n Testing with {len(list(keypoint_annotations))} "
                f"SemanticSegmentationAnnotation \n \n "
            )
            for keypoint_annotation in keypoint_annotations:
                assert isinstance(keypoint_annotation, KeypointAnnotation)
