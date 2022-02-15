from unity_vision.consumers.solo.parser import SoloBase
from unity_vision.protos.solo_pb2 import (BoundingBox2DAnnotation,
                                          BoundingBox3DAnnotation,
                                          InstanceSegmentationAnnotation,
                                          KeypointAnnotation, RGBCamera,
                                          SemanticSegmentationAnnotation)

__SENSORS__ = [
    {
        "sensor": RGBCamera,
        "annotations": [
            BoundingBox2DAnnotation,
            BoundingBox3DAnnotation,
            InstanceSegmentationAnnotation,
            SemanticSegmentationAnnotation,
            KeypointAnnotation,
        ],
    }
]


def test__init_sensor_pool():
    sensor_pool = SoloBase._init_sensor_pool()
    assert len(sensor_pool.keys()) > 0
    assert "unity.solo.RGBCamera" in list(sensor_pool.keys())
