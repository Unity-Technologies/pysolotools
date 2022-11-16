import glob
import logging
import os
import time

from pysolotools.core.models import (
    BoundingBox2DAnnotation,
    BoundingBox3DAnnotation,
    DatasetMetadata,
    Frame,
    InstanceSegmentationAnnotation,
    SemanticSegmentationAnnotation,
)

logger = logging.getLogger(__name__)


class FramesIterator:
    SENSORS = [
        {
            "sensor": "type.unity.com/unity.solo.RGBCamera",
            "annotations": [
                BoundingBox2DAnnotation,
                BoundingBox3DAnnotation,
                InstanceSegmentationAnnotation,
                SemanticSegmentationAnnotation,
            ],
        }
    ]

    def __init__(
        self,
        data_path: str,
        metadata: DatasetMetadata,
        start: int = 0,
        end: int = None,
    ):
        """
        Constructor for an Iterator that loads a Solo Frame from a Solo Dataset.

        Args:
            data_path (str): Path to dataset. This should have all sequences.
            metadata (DatasetMetadata): DatasetMetadata object
            start (int): Start sequence
            end (int): End sequence


        """
        super().__init__()
        self.frame_pool = list()
        self.data_path = os.path.normpath(data_path)
        self.frame_idx = start
        pre = time.time()
        self.metadata = metadata
        logger.info("DONE (t={:0.5f}s)".format(time.time() - pre))

        self.total_frames = self.metadata.totalFrames
        self.total_sequences = self.metadata.totalSequences
        self.steps_per_sequence = int(self.total_frames / self.total_sequences)

        self.end = end or self.__len__()

    def parse_frame(self, f_path: str) -> Frame:
        """
        Parses a json file to a pysolo Frame model.

        Args:
            f_path (str): Path to a step in a sequence for a frame.

        Returns:
            Frame:

        """
        with open(f_path, "r") as f:
            frame = Frame.from_json(f.read())
            return frame

    def __iter__(self):
        self.frame_idx = 0
        return self

    def __next__(self):
        if self.frame_idx >= self.end:
            raise StopIteration
        return self.__load_frame__(self.frame_idx)

    def __len__(self):
        return self.total_frames

    def __load_frame__(self, frame_id: int) -> Frame:
        sequence = int(frame_id / self.steps_per_sequence)
        step = frame_id % self.steps_per_sequence
        self.sequence_path = f"{self.data_path}/*sequence.{sequence}"
        filename_pattern = f"{self.sequence_path}/step{step}.frame_data.json"
        files = glob.glob(filename_pattern)
        # There should be exactly 1 frame_data for a particular sequence.
        if len(files) != 1:
            raise Exception(f"Metadata file not found for sequence {sequence}")
        self.frame_idx += 1
        return self.parse_frame(files[0])
