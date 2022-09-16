import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, List

from pysolotools.core.models import Frame

logger = logging.getLogger(__name__)


class AnalyzerBase(ABC):
    """
    Abstract class defining the essential interfaces to compute the stats.
    """

    @abstractmethod
    def analyze(self, frame: Frame = None, cat_ids: List = None, **kwargs: Any) -> Any:
        """
        Returns computed stats values.
        Args:
            frame (object): frame object.
            cat_ids(list): list of categories

        Returns:
            computed stats values.

        """
        pass

    @abstractmethod
    def merge(self, agg_result: Any, frame_result: Any, **kwargs: Any) -> Any:
        """
        Merge computed stats values.
        Args:
            agg_result (object): aggregated results.
            frame_result (object):  result of one frame.

        Returns:
            aggregated stats values.

        """
        pass


class AnalyzerFactory:
    """Factory class for creating Analyzers with a dynamic registry."""

    registry = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """Class method to register analyzers"""

        def wrapper(analyzer_class: AnalyzerBase) -> Callable:
            if name in cls.registry:
                logger.warning("Analyzer %s already exists. Will replace it", name)
            cls.registry[name] = analyzer_class
            return analyzer_class

        return wrapper

    @classmethod
    def create_analyzer(cls, name: str, **kwargs) -> AnalyzerBase:
        """Class method to create ana analyzers instance"""

        if name not in cls.registry:
            logger.warning("Analyzer %s does not exist in the registry", name)
            return None

        analyzer_class = cls.registry[name]
        analyzer = analyzer_class(**kwargs)
        return analyzer
