from typing import Any

from pysolotools.stats.analyzers.base import StatsAnalyzer


class BBoxSizeAnalyzer(StatsAnalyzer):
    def analyze(self, frame: object = None, **kwargs: Any) -> object:

        print("Compute BBoxSizeAnalyzer stats")
