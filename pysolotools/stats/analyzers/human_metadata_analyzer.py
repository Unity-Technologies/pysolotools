import dataclasses
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

from pysolotools.core.models import Frame
from pysolotools.core.models.solo import HumanMetadataAnnotation, HumanMetadataLabel
from pysolotools.stats.analyzers.base import StatsAnalyzer


@dataclass
class AggregateObject:
    key: str
    value: Union[str, int, float, None] = None
    sum: Union[None, float] = None
    count: int = 0

    def add(self, value: Union[str, int, float]) -> "AggregateObject":
        self.value = None
        self.count += 1
        if isinstance(value, str):
            return self
        self.sum = value if not self.sum else self.sum + value
        return self


@dataclass
class HumanMetadataAnnotationAggregate:
    statistics: Dict[str, AggregateObject] = field(
        default_factory=lambda: defaultdict(dict)
    )

    def to_dict(self) -> dict:
        result = {}
        for k, v in self.statistics.items():
            result[k] = dataclasses.asdict(v)
        return result


class HumanMetadataAnnotationAnalyzer(StatsAnalyzer):
    def __init__(self):
        self._res: HumanMetadataAnnotationAggregate = HumanMetadataAnnotationAggregate()

    @staticmethod
    def _filter(frame: Frame) -> List[HumanMetadataAnnotation]:
        results = []

        for capture in frame.captures:
            results.extend(
                filter(
                    lambda k: isinstance(k, HumanMetadataAnnotation),
                    capture.annotations,
                )
            )
        return results

    def analyze(self, frame: Frame = None, **kwargs: Any) -> List[AggregateObject]:
        """
        Args:
            frame (Frame): metadata of one frame
        Returns:

        """
        human_metadata = HumanMetadataAnnotationAnalyzer._filter(frame=frame)
        human_metadata_stats = []
        for metadata in human_metadata:
            person: HumanMetadataLabel
            for person in metadata.metadata:
                human_metadata_stats.append(AggregateObject(key="human"))
                human_metadata_stats.append(AggregateObject(key=person.age))
                human_metadata_stats.append(AggregateObject(key=person.sex))
                human_metadata_stats.append(AggregateObject(key=person.ethnicity))

        return human_metadata_stats

    def merge(self, frame_result: List[AggregateObject], **kwargs):
        """
        Merge computed stats values.
        Args:
            frame_result (list):  result of one frame.

        Returns:

        """
        for frame in frame_result:
            self._res.statistics.setdefault(frame.key, frame).add(value=frame.value)

    def get_result(self) -> dict:
        return self._res.to_dict()
