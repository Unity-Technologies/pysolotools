import datetime
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class AnnotationLabel:
    instanceId: int
    labelId: int


@dataclass(frozen=True)
class Annotation:
    id: str
    sensorId: str
    description: str


@dataclass(frozen=True)
class BoundingBox2DLabel(AnnotationLabel):
    labelName: str
    origin: List[float]
    dimension: List[float]


@dataclass(frozen=True)
class BoundingBox3DLabel(AnnotationLabel):
    size: int
    translation: List[float]
    rotation: List[float]


@dataclass(frozen=True)
class Keypoint:
    index: int
    location: List[float]
    state: int


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class Capture:
    id: str
    description: str
    position: List[float]
    rotation: List[float]
    annotations: List[Annotation]


@dataclass(frozen=True)
class Frame:
    frame: int
    sequence: int
    step: int
    captures: List[Capture]


@dataclass(frozen=True)
class Sensor:
    id: str
    description: str
    annotations: List[Annotation]


@dataclass(frozen=True)
class RGBCameraCapture(Capture):
    position: List[float]
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
