import glob
import os
import time

from pysolo.core.models import (
    BoundingBox2DAnnotation,
    BoundingBox3DAnnotation,
    DatasetAnnotations,
    DatasetMetadata,
    Frame,
    InstanceSegmentationAnnotation,
    SemanticSegmentationAnnotation,
)


class FramesIterator:
    """
    Parser for solo dataset with 1 frame per sequence. Returns a list of frames from each sequence.
    Essentially flattens the sequence.
    """

    """
        Map of sensor to its relevant annotations available for the sensor.
        """
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

    def __init__(self,
                 data_path: str,
                 metadata: DatasetMetadata,
                 annotation_definitions: DatasetAnnotations,
                 start: int = 0,
                 end: int = None):
        """

        Args:
            data_path (str): Path to dataset. This should have all sequences.
            metadata (DatasetMetadata): DatasetMetadata object
            start (int): Start sequence
            end (int): End sequence
            metadata_path (str): Optional kwarg for providing custom metadata json file path.

        Example:
        └── solo
            ├── annotation_definitions.json
            ├── metadata.json
            ├── metric_definitions.json
            ├── sensor_definitions.json
            ├── sequence.0
            │         ├── step0.camera.png
            │         └── step0.frame_data.json
            ├── sequence.1
            │         ├── step0.camera.png
            │         └── step0.frame_data.json
            ├── sequence.2
            │         ├── step0.camera.png
            │         └── step0.frame_data.json

        """
        super().__init__()
        self.frame_pool = list()
        self.data_path = os.path.normpath(data_path)
        self.frame_idx = start
        pre = time.time()
        self.metadata = metadata
        self.annotation_definitions = annotation_definitions
        print('DONE (t={:0.5f}s)'.format(time.time() - pre))

        self.total_frames = self.metadata.totalFrames
        self.total_sequences = self.metadata.totalSequences
        self.steps_per_sequence = int(self.total_frames / self.total_sequences)

        self.end = end or self.__len__()

    def __load_frame__(self, frame_id: int) -> Frame:
        """

        Args:
            frame_id (int): Frame id in the sequence.

        Returns:

        """
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

    def parse_frame(self, f_path: str) -> Frame:
        """

        Args:
            f_path (str): Path to a step in a sequence for a frame.

        Returns:
            Frame: Returns a Frame object.

        """
        f = open(f_path, "r")
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
