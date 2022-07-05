from pysolotools.core.models import DatasetMetadata

_mock_dataset_metadata_all = {
    "unityVersion": "2020.3.34f1",
    "perceptionVersion": "0.10.0-preview.1",
    "renderPipeline": "HDRP",
    "simulationStartTime": "5/20/2022 1:54:42 PM",
    "scenarioRandomSeed": 539662031,
    "scenarioActiveRandomizers": ["AnimationRandomizer"],
    "totalFrames": 1,
    "totalSequences": 1,
    "sensors": ["camera"],
    "metricCollectors": [],
    "simulationEndTime": "5/20/2022 1:55:40 PM",
    "annotators": [
        {
            "name": "bounding box",
            "type": "type.unity.com/unity.solo.BoundingBox2DAnnotation",
        }
    ],
}

__dataset_metadata_optionals = [
    "simulationStartTime",
    "simulationEndTime",
    "renderPipeline",
    "scenarioRandomSeed",
]


def test_dataset_metadata_all():
    dataset_metadata = DatasetMetadata.from_dict(_mock_dataset_metadata_all)
    assert dataset_metadata


def test_dataset_metadata_optionals():
    for key in __dataset_metadata_optionals:
        m = _mock_dataset_metadata_all
        m.pop(key)
        dataset_metadata = DatasetMetadata.from_dict(m)
        assert dataset_metadata
