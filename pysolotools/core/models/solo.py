from dataclasses import dataclass, field
from typing import List

import pandas as pd
from dataclasses_json import config, dataclass_json

from pysolotools.core.exceptions import MissingCaptureException


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
    labelName: str
    size: List[float]
    translation: List[float]
    rotation: List[float]
    velocity: List[float]
    acceleration: List[float]


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

    def get_file_path(self, capture: Capture) -> str:
        """
        Returns:
            str: File path of given capture having current sequence as root
        """
        c = list(filter(lambda cap: isinstance(cap, capture), self.captures))
        if not c:
            raise MissingCaptureException(f"{capture.__name__} is missing.")

        return f"sequence.{self.sequence}/{c[0].filename}"

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
class KeypointTemplateDefinition:
    templateId: str
    templateName: str
    keypoints: List[KeypointDefinition]


@dataclass
class KeypointAnnotationDefinition(AnnotationDefinition):
    template: KeypointTemplateDefinition


@dataclass
class LabelNameSpec:
    label_id: int
    label_name: str


@dataclass
class BoundingBox2DAnnotationDefinition(AnnotationDefinition):
    spec: List[LabelNameSpec]


@dataclass
class SemanticSegmentationAnnotationDefinition(AnnotationDefinition):
    pass  # Adds not additional fields


@dataclass
class BoundingBox3DAnnotationDefinition(AnnotationDefinition):
    spec: List[LabelNameSpec]


@dataclass
class InstanceSegmentationAnnotationDefinition(AnnotationDefinition):
    spec: List[LabelNameSpec]


@dataclass
class BoundingBoxAnnotationDefinitionSpec:
    label_id: int
    label_name: str


@dataclass
class BoundingBoxAnnotationDefinition:
    type: str
    id: str
    description: str
    spec: List[BoundingBoxAnnotationDefinitionSpec]

    def __post_init__(self):
        cat_id_to_name = {}
        cat_name_to_id = {}

        for s in self.spec:
            cat_id_to_name[s.label_id] = s.label_name
            cat_name_to_id[s.label_name] = s.label_id

    def get_cat_ids(self) -> List[int]:
        return [s.label_id for s in self.spec]


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


@dataclass_json
@dataclass
class DatasetAnnotations(object):
    annotationDefinitions: List[dataclass]

    def __post_init__(self):
        self.annotationDefinitions = [
            DefinitionFactory.cast(anno) for anno in self.annotationDefinitions
        ]


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
        if "@type" not in data.keys():
            raise Exception("No type provided in annotation")
        dtype = data["@type"]
        if dtype not in cls.switcher.keys():
            raise Exception("Unknown data type")
        klass = cls.switcher[dtype]
        return klass.from_dict(data)


class DefinitionFactory:
    switcher = {
        "type.unity.com/unity.solo.KeypointAnnotation": KeypointAnnotationDefinition,
        "type.unity.com/unity.solo.BoundingBox2DAnnotation": BoundingBox2DAnnotationDefinition,
        "type.unity.com/unity.solo.BoundingBox3DAnnotation": BoundingBox3DAnnotationDefinition,
        "type.unity.com/unity.solo.SemanticSegmentationAnnotation": SemanticSegmentationAnnotationDefinition,
        "type.unity.com/unity.solo.InstanceSegmentationAnnotation": InstanceSegmentationAnnotationDefinition,
    }

    @classmethod
    def cast(cls, data):
        if "@type" not in data.keys():
            raise Exception("No type provided in annotation")
        dtype = data["@type"]
        if dtype not in cls.switcher.keys():
            raise Exception(f"Unknown data type: {dtype}")
        klass = cls.switcher[dtype]
        return klass.from_dict(data)
