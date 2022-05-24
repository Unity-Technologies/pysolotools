import glob

from pysolo.core import DatasetMetadata
from pysolo.core.iterators import FramesIterator
from pysolo.core.models import DatasetAnnotations
from pysolo.core.stats import SoloStats


class Solo:
    def __init__(self, data_path: str, metadata_file: str = None, start: int = 0, end: int = None):
        self.data_path = data_path
        # self.metadata_file = metadata_file
        self.start = start
        self.end = end

        self.metadata = self.__open_metadata__(metadata_file)
        self.annotation_definitions = self.__open_annotation_definitions__()

    def frames(self) -> FramesIterator:
        return FramesIterator(
            self.data_path,
            self.metadata,
            self.annotation_definitions,
            self.start,
            self.end
        )

    def stats(self) -> SoloStats:
        return SoloStats(self.frames())

    def get_metadata(self) -> DatasetMetadata:
        """
        Returns:
            pysolo.core.models.DatasetMetadata: Returns metadata of SOLO Dataset
        """
        return self.metadata

    def get_annotation_definitions(self) -> DatasetAnnotations:
        return self.annotation_definitions

    def __open_metadata__(self, metadata_file: str = None) -> DatasetMetadata:
        """
        Default metadata location is expected at root/metadata.json but
        if an annotation_file path is provided that is used as the annotation

        Metadata can be in one of two locations, depending if it was a part of a singular build,
        or if it was a part of a distributed build.

        Args:
            metadata_file (str): Path to solo annotation file



        """
        if metadata_file:
            discovered_path = [metadata_file]
        else:
            discovered_path = glob.glob(self.data_path + "/metadata.json", recursive=True)
            if len(discovered_path) != 1:
                raise Exception("Found none or multiple metadata files.")
        metadata_f = open(discovered_path[0])
        return DatasetMetadata.from_json(metadata_f.read())

    def __open_annotation_definitions__(self) -> DatasetAnnotations:
        discovered_path = glob.glob(
            self.data_path + "/annotation_definitions.json", recursive=True)
        if len(discovered_path) != 1:
            raise Exception(
                "Found none or multiple annotation definition files.")
        metadata_f = open(discovered_path[0])
        return DatasetAnnotations.from_json(metadata_f.read())
