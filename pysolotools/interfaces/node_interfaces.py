from abc import ABC, abstractmethod
from typing import Any

from pysolotools.core.models.solo import Frame


class Node(ABC):
    """
    Abstract class defining the essential interfaces for a pysolo post-processing
    plugin tool.
    """

    @abstractmethod
    def get_id(self) -> str:
        """
        Returns a unique ID for plugin.

        Returns:
            A Unique ID for the plugin.

        """
        pass


class ConverterNode(Node):
    """
    Abstract class defining the essential interfaces for a converter node. A
    converter node is a post-processing tool that converts SOLO data into an
    alternative format.
    """

    @abstractmethod
    def get_id(self) -> str:
        """
        Returns a unique ID for plugin.

        Returns:
            A Unique ID for the plugin.

        """
        pass

    @abstractmethod
    def convert(self, **kwargs: Any):
        """
        Converter node used to convert a Solo dataset into an alternate format or vice versa
        """
        pass


class TransformerNode(Node):
    """
    Abstract class defining the essential interfaces for a transformer node. A
    transformer node is a post-processing tool modifies the solo dataset before
    conversion of the data. Examples of forseen modification could be filtering
    of the data, changing values of the data (i.e. buffering out bounding boxes,
    clipping images), and augmentation of the data (adding fields to records)
    """

    @abstractmethod
    def get_id(self) -> str:
        """
        Returns a unique ID for plugin.
        Returns:
            A Unique ID for the plugin.

        """
        pass

    @abstractmethod
    def transform(self, frame: Frame, **kwargs: Any) -> Frame:
        """
        Transformer node that modifies/transforms a Solo frame of data
        Args:
            frame (Frame): The Solo frame

        Returns:
            The modified frame

        """
        pass


class AnalyzerNode(Node):
    """
    Abstract class defining the essential interfaces for an analyzer node. An
    analyzer node is a post-processing tool that calculates statistics on the
    dataset.
    """

    @abstractmethod
    def get_id(self) -> str:
        """
        Returns a unique ID for plugin.
        Returns:
            A Unique ID for the plugin.

        """
        pass

    @abstractmethod

    def analyze(self, frame: Frame, **kwargs: Any) -> Frame:
        """
        Analyze node that analyzes a Solo frame of data
        Args:
            frame (Frame): The Solo frame
        Returns:
            The solo frame

        """
        pass

    @abstractmethod
    def aggregate(self, arguments: Dict[str, Any]) -> None:
        pass
