from dataclasses import dataclass, field
from typing import List

import pandas as pd
from dataclasses_json import config, dataclass_json


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
    velocity: List[float]
    acceleration: List[float]
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


@dataclass
class AnnotationDefinition:
    annotation_definitions: List[dataclass]

    def __post_init__(self):
        self.annotation_definitions = [DataFactory.cast(anno_def) for anno_def in self.annotation_definitions]


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


@dataclass
class SoloDataset:
    metadata: DatasetMetadata
    annotation_definitions: List[dataclass]

    def __post_init__(self):
        self.annotation_definitions = [DataFactory.cast(anno) for anno in self.annotation_definitions]

    def get_num_frames(self):
        """

        Returns:

        """
        return self.metadata.totalFrames


class DataFactory:
    switcher = {
        "type.unity.com/unity.solo.RGBCamera": RGBCameraCapture,
        "type.unity.com/unity.solo.KeypointAnnotation": KeypointAnnotation,
        "type.unity.com/unity.solo.BoundingBox2DAnnotation": BoundingBox2DAnnotation,
        "type.unity.com/unity.solo.BoundingBox3DAnnotation": BoundingBox3DAnnotation,
        "type.unity.com/unity.solo.InstanceSegmentationAnnotation": InstanceSegmentationAnnotation,
        "type.unity.com/unity.solo.SemanticSegmentationAnnotation": SemanticSegmentationAnnotation,
        "type.unity.com/unity.solo.BoundingBoxAnnotationDefinition": BoundingBoxAnnotationDefinition

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
