from pysolotools.consumers import Solo
from pysolotools.stats.analyzers.human_annotation_analyzer import (
    HumanMetadataAnnotationAnalyzer,
)
from pysolotools.stats.handler import StatsHandler


def test_foo():
    def print_thing(key: str, source: dict):
        print(f"{key}: {source.get(key, {}).get('count', 0)}")

    solo = Solo(
        data_path="/Users/drew.flintosh/Library/Application Support/DefaultCompany/test-palette-2/solo"
    )
    handler = StatsHandler(solo=solo)
    analyzers = [HumanMetadataAnnotationAnalyzer()]
    result = handler.handle(analyzers=analyzers)
    analyzer = result.get("HumanMetadataAnnotationAnalyzer")
    things_i_care_about = [
        "human",
        "Adult",
        "Toddler",
        "Female",
        "Male",
        "African",
        "Asian",
        "Caucasian",
        "LatinAmerican",
        "MiddleEastern",
    ]
    list(map(lambda x: print_thing(x, analyzer), things_i_care_about))

    for k, v in analyzer.items():
        if v.get("values", None):
            print(k)
