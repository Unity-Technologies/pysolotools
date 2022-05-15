import datetime
from dataclasses import dataclass
from typing import List

from dataclasses_json import dataclass_json


@dataclass(frozen=True)
class AnnotationLabel:
    instanceId: int
    labelId: int


@dataclass_json
@dataclass(frozen=True)
class Annotation:
    id: str
    sensorId: str
    description: str


@dataclass_json
@dataclass(frozen=True)
class BoundingBox2DLabel(AnnotationLabel):
    labelName: str
    origin: List[float]
    dimension: List[float]


@dataclass_json
@dataclass(frozen=True)
class BoundingBox3DLabel(AnnotationLabel):
    size: List[float]
    translation: List[float]
    rotation: List[float]


@dataclass(frozen=True)
class Keypoint:
    index: int
    location: List[float]
    state: int


class KeypointLabel(AnnotationLabel):
    pose: str
    keypoints: List[Keypoint]


@dataclass(frozen=True)
class InstanceSegmentationLabel(AnnotationLabel):
    labelName: str
    color: List[int]


@dataclass(frozen=True)
class SemanticSegmentationLabel(AnnotationLabel):
    labelName: str
    pixelValue: List[int]


@dataclass(frozen=True)
class KeypointAnnotation(Annotation):
    templateId: str
    values: List[KeypointLabel]


@dataclass(frozen=True)
class BoundingBox2DAnnotation(Annotation):
    values: List[BoundingBox2DLabel]


@dataclass(frozen=True)
class BoundingBox3DAnnotation(Annotation):
    values: List[BoundingBox3DLabel]


@dataclass(frozen=True)
class InstanceSegmentationAnnotation(Annotation):
    imageFormat: str
    dimension: List[int]
    filename: str
    instances: List[InstanceSegmentationLabel]


@dataclass(frozen=True)
class SemanticSegmentationAnnotation(Annotation):
    imageFormat: str
    dimension: List[int]
    filename: str
    instances: List[SemanticSegmentationLabel]


@dataclass_json
@dataclass
class Capture:
    """
    id (str): Id
    type(str):
    """
    id: str
    type: str
    description: str
    position: List[float]
    rotation: List[float]
    annotations: List[dataclass]

    def __post_init__(self):
        self.annotations = [DataFactory.cast(anno) for anno in self.annotations]

    def __eq__(self, other):
        if other == self.id:
            return True
        return False


@dataclass_json
@dataclass
class Frame:
    frame: int
    sequence: int
    step: int
    captures: List[dataclass]
    metrics: List[object]

    def __post_init__(self):
        self.captures = [DataFactory.cast(capture) for capture in self.captures]


@dataclass(frozen=True)
class Sensor:
    id: str
    description: str
    annotations: List[Annotation]


@dataclass
class RGBCameraCapture(Capture):
    rotation: List[float]
    velocity: List[float]
    acceleration: List[float]
    filename: str
    imageFormat: str
    dimension: List[float]
    projection: str
    matrix: List[float]


@dataclass(frozen=True)
class Dataset:
    id: str
    name: str
    createdAt: datetime.date = None
    updatedAt: datetime.date = None
    description: str = None
    licenseURI: str = None


@dataclass(frozen=True)
class Archive(object):
    id: str
    name: str
    type: str
    state: dict = None
    downloadURL: str = None
    uploadURL: str = None
    createdAt: datetime.date = None
    updatedAt: datetime.date = None


@dataclass(frozen=True)
class Attachment(object):
    id: str
    name: str
    description: str = None
    state: dict = None
    downloadURL: str = None
    uploadURL: str = None
    createdAt: datetime.date = None
    updatedAt: datetime.date = None


class DataFactory:
    switcher = {
        "type.unity.com/unity.solo.RGBCamera": RGBCameraCapture,
        "type.unity.com/unity.solo.KeypointAnnotation": KeypointAnnotation,
        "type.unity.com/unity.solo.BoundingBox2DAnnotation": BoundingBox2DAnnotation,
        "type.unity.com/unity.solo.BoundingBox3DAnnotation": BoundingBox3DAnnotation,
        "type.unity.com/unity.solo.InstanceSegmentationAnnotation": InstanceSegmentationAnnotation,
        "type.unity.com/unity.solo.SemanticSegmentationAnnotation": SemanticSegmentationAnnotation

    }

    @classmethod
    def cast(cls, data):
        if 'type' not in data.keys():
            raise Exception("No type provided in annotation")
        dtype = data['type']
        if dtype not in cls.switcher.keys():
            raise Exception("Unknown data type")
        klass = cls.switcher[dtype]
        return klass.from_dict(data)
