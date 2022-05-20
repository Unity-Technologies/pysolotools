import datetime
from dataclasses import dataclass, field
from typing import List

import pandas as pd
from dataclasses_json import config, dataclass_json


@dataclass
class AnnotationLabel:
    instanceId: int
    labelId: int


@dataclass_json
@dataclass
class Annotation:
    type: str = field(metadata=config(field_name="@type"))
    id: str
    sensorId: str
    description: str


@dataclass_json
@dataclass
class BoundingBox2DLabel(AnnotationLabel):
    labelName: str
    origin: List[float]
    dimension: List[float]


@dataclass_json
@dataclass
class BoundingBox3DLabel(AnnotationLabel):
    size: List[float]
    translation: List[float]
    rotation: List[float]


@dataclass
class Keypoint:
    index: int
    location: List[float]
    state: int


@dataclass
class KeypointLabel(AnnotationLabel):
    pose: str
    keypoints: List[Keypoint]


@dataclass
class InstanceSegmentationLabel(AnnotationLabel):
    labelName: str
    color: List[int]


@dataclass
class SemanticSegmentationLabel:
    labelName: str
    pixelValue: List[int]


@dataclass
class KeypointAnnotation(Annotation):
    templateId: str
    values: List[KeypointLabel]


@dataclass
class BoundingBox2DAnnotation(Annotation):
    values: List[BoundingBox2DLabel]


@dataclass
class BoundingBox3DAnnotation(Annotation):
    velocity: List[float]
    acceleration: List[float]
    values: List[BoundingBox3DLabel]


@dataclass
class InstanceSegmentationAnnotation(Annotation):
    imageFormat: str
    dimension: List[int]
    filename: str
    instances: List[InstanceSegmentationLabel]


@dataclass
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
    type: str = field(metadata=config(field_name="@type"))
    description: str
    position: List[float]
    rotation: List[float]
    annotations: List[dataclass]

    def get_annotations_df(self) -> pd.DataFrame:
        """
        Returns:
                pd.DataFrame: Captures List fo
        """
        return pd.DataFrame(self.annotations)

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

    def get_captures_df(self) -> pd.DataFrame:
        """
        Returns:
                pd.DataFrame: Captures List for a Solo Frame
        """
        return pd.DataFrame(self.captures)

    def get_metrics_df(self) -> pd.DataFrame:
        """

        Returns:
            pd.DataFrame: Solo Frame Metrics
        """
        return pd.DataFrame(self.metrics)


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


@dataclass_json
@dataclass
class DatasetMetadata(object):
    unityVersion: str
    perceptionVersion: str
    renderPipeline: str
    simulationStartTime: str
    scenarioRandomSeed: float
    scenarioActiveRandomizers: List[str]
    totalFrames: int
    totalSequences: int
    sensors: List[str]
    metricCollectors: List[str]
    simulationEndTime: str
    annotators: List[object]

@dataclass_json
@dataclass
class AnnotationDefinition:
    id: str
    description: str

@dataclass
class KeypointDefinition:
    label: str
    index: int
    color: List[int]

@dataclass
class KeypointAnnotationDefinition(AnnotationDefinition):
    templateId: str
    templateName: str
    keypoints: List[KeypointDefinition]

@dataclass
class BoundingBox2DSpec:
    label_id: int
    label_name: str

@dataclass
class BoundingBox2DAnnotationDefinition(AnnotationDefinition):
    spec: List[BoundingBox2DSpec]

@dataclass
class SemanticSegmentationAnnotationDefinition(AnnotationDefinition):
    pass

@dataclass_json
@dataclass
class DatasetAnnotations(object):
    annotationDefinitions: List[dataclass]

    def __post_init__(self):
        self.annotationDefinitions = [DefinitionFactory.cast(anno) for anno in self.annotationDefinitions]

class DataFactory:
    switcher = {
        "type.unity.com/unity.solo.RGBCamera": RGBCameraCapture,
        "type.unity.com/unity.solo.KeypointAnnotation": KeypointAnnotation,
        "type.unity.com/unity.solo.BoundingBox2DAnnotation": BoundingBox2DAnnotation,
        "type.unity.com/unity.solo.BoundingBox3DAnnotation": BoundingBox3DAnnotation,
        "type.unity.com/unity.solo.InstanceSegmentationAnnotation": InstanceSegmentationAnnotation,
        "type.unity.com/unity.solo.SemanticSegmentationAnnotation": SemanticSegmentationAnnotation,

    }

    @classmethod
    def cast(cls, data):
        if '@type' not in data.keys():
            raise Exception("No type provided in annotation")
        dtype = data['@type']
        if dtype not in cls.switcher.keys():
            raise Exception("Unknown data type")
        klass = cls.switcher[dtype]
        return klass.from_dict(data)

class DefinitionFactory:
    switcher = {
        "type.unity.com/unity.solo.KeypointAnnotation": KeypointAnnotationDefinition,
        "type.unity.com/unity.solo.BoundingBox2DAnnotation": BoundingBox2DAnnotationDefinition,
        "type.unity.com/unity.solo.SemanticSegmentationAnnotation": SemanticSegmentationAnnotationDefinition,
    }

    @classmethod
    def cast(cls, data):
        if '@type' not in data.keys():
            raise Exception("No type provided in annotation")
        dtype = data['@type']
        if dtype not in cls.switcher.keys():
            raise Exception("Unknown data type")
        klass = cls.switcher[dtype]
        return klass.from_dict(data)