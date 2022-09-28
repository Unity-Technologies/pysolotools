import dataclasses
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

from pysolotools.core.models import Frame
from pysolotools.core.models.solo import HumanMetadataAnnotation, HumanMetadataLabel
from pysolotools.stats.analyzers.base import StatsAnalyzer


@dataclass
class AggregateObject:
    sum: float = None
    count: int = 0
    max: float = None
    min: float = None

    def add(self, value: Union[str, int, float]) -> "AggregateObject":
        self.count += 1
        if isinstance(value, str):
            return self
        self.sum = value if not self.sum else self.sum + value
        self.max = value if not self.max else max(self.max, value)
        self.min = value if not self.min else min(self.min, value)
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

    def _add_statistic(self, key: str, value: Union[str, int, float]):
        self._res.statistics.setdefault(key, AggregateObject().add(value=value)).add(
            value=value
        )

    def analyze(
        self, frame: Frame = None, **kwargs: Any
    ) -> HumanMetadataAnnotationAggregate:
        """
        Args:
            frame (Frame): metadata of one frame
        Returns:

        """
        human_metadata = HumanMetadataAnnotationAnalyzer._filter(frame=frame)
        for metadata in human_metadata:
            person: HumanMetadataLabel
            for person in metadata.metadata:
                self._add_statistic(key="human", value=1)
                self._add_statistic(key=person.age, value=person.age)
                self._add_statistic(key="height", value=float(person.height))
                self._add_statistic(key="weight", value=float(person.weight))
                self._add_statistic(key=person.sex, value=person.sex)
                self._add_statistic(key=person.ethnicity, value=person.ethnicity)

        return self._res

    def merge(self, frame_result: List, **kwargs):
        """
        Merge computed stats values.
        Args:
            frame_result (list):  result of one frame.

        Returns:
            aggregated stats values.

        """
        pass

    def get_result(self) -> dict:
        return self._res.to_dict()
