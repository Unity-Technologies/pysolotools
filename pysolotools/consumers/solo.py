import glob
import logging
from typing import Dict

from pysolotools.core import BoundingBox2DAnnotationDefinition, DatasetMetadata
from pysolotools.core.iterators import FramesIterator
from pysolotools.core.models import (
    BoundingBox3DAnnotationDefinition,
    DatasetAnnotations,
    InstanceSegmentationAnnotationDefinition,
)

"""
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

logger = logging.getLogger(__name__)


class Solo:
    def __init__(
        self,
        data_path: str,
        metadata_file: str = None,
        annotation_definitions_file: str = None,
        start: int = 0,
        end: int = None,
        **kwargs
    ):
        """
        Constructor for Unity SOLO helper class.
        Args:
            data_path (str): Location for the root folder of Solo dataset
            metadata_file (str): Location for the metadata.json file for the Solo dataset
            start (Optional[int]): Start index for frames in the dataset
            end (Optional[int]): End index for frames in the dataset
        """

        self.data_path = data_path
        self.start = start
        self.end = end

        self.metadata = self.__open_metadata__(metadata_file)
        self.annotation_definitions = self.__open_annotation_definitions__(
            annotation_definitions_file
        )

    def frames(self) -> FramesIterator:
        """
        Return a Frames Iterator

        Returns:
            FramesIterator
        """
        return FramesIterator(
            self.data_path,
            self.metadata,
            self.start,
            self.end,
        )

    def categories(self) -> Dict[int, str]:
        categories = {}
        for d in self.annotation_definitions.annotationDefinitions:
            if isinstance(d, BoundingBox2DAnnotationDefinition):
                for s in d.spec:
                    categories[s.label_id] = s.label_name
                return categories
            elif isinstance(d, BoundingBox3DAnnotationDefinition):
                for s in d.spec:
                    categories[s.label_id] = s.label_name
                return categories
            elif isinstance(d, InstanceSegmentationAnnotationDefinition):
                for s in d.spec:
                    categories[s.label_id] = s.label_name
                return categories
        return None

    def frame_ids(self):
        return list(map(lambda f: f.frame, self.frames()))

    def get_metadata(self) -> DatasetMetadata:
        """
        Get metadata for the Solo dataset

        Returns:
            DatasetMetadata: Returns metadata of SOLO Dataset
        """
        return self.metadata

    def get_annotation_definitions(self) -> DatasetAnnotations:
        """
        Get annotation definitions for the Solo dataset

        Returns:
            DatasetAnnotations

        """
        return self.annotation_definitions

    def __open_metadata__(self, metadata_file: str = None) -> DatasetMetadata:
        """
        Default metadata location is expected at root/metadata.json but
        if an metadata_file path is provided that is used as the metadata file path.

        Metadata can be in one of two locations, depending if it was a part of a singular build,
        or if it was a part of a distributed build.

        Args:
            metadata_file (str): Path to solo annotation file

        Returns:
            DatasetMetadata

        """
        if metadata_file:
            discovered_path = [metadata_file]
        else:
            discovered_path = glob.glob(
                self.data_path + "/metadata.json", recursive=True
            )
            if len(discovered_path) != 1:
                raise Exception("Found none or multiple metadata files.")

        with open(discovered_path[0]) as metadata_f:
            return DatasetMetadata.from_json(metadata_f.read())

    def __open_annotation_definitions__(
        self, annotation_definitions_file: str = None
    ) -> DatasetAnnotations:
        """
        Default annotation_definitions.json is expected in the root folder of the Solo dataset. If a custom
        `annotation_definitions_file` is provided then that is used instead.

        Args:
            annotation_definitions_file (str): Custom path for annotation_definitions.json file

        Returns:
            DatasetAnnotations

        """
        if annotation_definitions_file:
            discovered_path = [annotation_definitions_file]
        else:
            discovered_path = glob.glob(
                self.data_path + "/annotation_definitions.json", recursive=True
            )
            if len(discovered_path) != 1:
                raise Exception("Found none or multiple annotation definition files.")
        with open(discovered_path[0]) as metadata_f:
            return DatasetAnnotations.from_json(metadata_f.read())
