from pysolotools.consumers import Solo
from pysolotools.stats.analyzers.human_annotation_analyzer import (
    HumanMetadataAnnotationAnalyzer,
)
from pysolotools.stats.handler import StatsHandler


def test_foo():
    solo = Solo(
        data_path="/Users/drew.flintosh/Library/Application Support/DefaultCompany/test-palette-2/solo"
    )
    handler = StatsHandler(solo=solo)
    analyzers = [HumanMetadataAnnotationAnalyzer()]
    result = handler.handle(analyzers=analyzers)
    print(result)
