import glob
import os
import time
from abc import ABC, abstractmethod

from pysolo.core.models import (BoundingBox2DAnnotation,
                                BoundingBox3DAnnotation, DatasetAnnotations,
                                DatasetMetadata, Frame,
                                InstanceSegmentationAnnotation,
                                SemanticSegmentationAnnotation)


class SoloBase(ABC):
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
                SemanticSegmentationAnnotation
            ]
        }
    ]

    def __init__(self):
        self.sensor_pool = self.SENSORS

    @abstractmethod
    def parse_frame(self, path) -> Frame:
        pass


class Solo(SoloBase):
    """
    Parser for solo dataset with 1 frame per sequence. Returns a list of frames from each sequence.
    Essentially flattens the sequence.
    """

    def __init__(self,
                 path: str,
                 annotation_file: str = None,
                 start: int = 0,
                 end: int = None,
                 *args,
                 **kwargs):
        """

        Args:
            path (str): Path to dataset. This should have all sequences.
            annotation_file (str): Location of annotation file
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
        self.path = os.path.normpath(path)
        self.frame_idx = start
        pre = time.time()
        self.metadata = self.__open_metadata__(annotation_file)
        self.annotation_definitions = self.__open_annotation_definitions__(annotation_file)
        print('DONE (t={:0.5f}s)'.format(time.time() - pre))

        self.total_frames = self.metadata.totalFrames
        self.total_sequences = self.metadata.totalSequences
        self.steps_per_sequence = int(self.total_frames / self.total_sequences)

        self.end = end or self.__len__()

    def get_metadata(self) -> DatasetMetadata:
        """
        Returns:
            pysolo.core.models.DatasetMetadata: Returns metadata of SOLO Dataset
        """
        return self.metadata

    def __open_metadata__(self, annotation_file: str = None):
        """
        Default metadata location is expected at root/metadata.json but
        if an annotation_file path is provided that is used as the annotation

        Metadata can be in one of two locations, depending if it was a part of a singular build,
        or if it was a part of a distributed build.

        Args:
            annotation_file (str): Path to solo annotation file



        """
        if annotation_file:
            discovered_path = [annotation_file]
        else:
            discovered_path = glob.glob(self.path + "/metadata.json", recursive=True)
            if len(discovered_path) != 1:
                raise Exception("Found none or multiple metadata files.")
        metadata_f = open(discovered_path[0])
        return DatasetMetadata.from_json(metadata_f.read())

    def get_annotation_definitions(self) -> DatasetAnnotations:
        return self.annotation_definitions

    def __open_annotation_definitions__(self, annotation_file: str = None):
        if annotation_file:
            discovered_path = [annotation_file]
        else:
            discovered_path = glob.glob(
                self.path + "/annotation_definitions.json", recursive=True)
            if len(discovered_path) != 1:
                raise Exception(
                    "Found none or multiple annottion definition files.")
        metadata_f = open(discovered_path[0])
        return DatasetAnnotations.from_json(metadata_f.read())

    def __load_frame__(self, frame_id: int) -> Frame:
        """

        Args:
            frame_id (int): Frame id in the sequence.

        Returns:

        """
        sequence = int(frame_id / self.steps_per_sequence)
        step = frame_id % self.steps_per_sequence
        self.sequence_path = f"{self.path}/*sequence.{sequence}"
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
