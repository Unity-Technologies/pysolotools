import logging
from dataclasses import dataclass, field
from typing import Callable, List

import pandas as pd
from dataclasses_json import CatchAll, Undefined, config, dataclass_json

from pysolotools.core.exceptions import MissingCaptureException

logger = logging.getLogger(__name__)


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
        self.annotations = [
            DataFactory.cast_annotation(anno)
            for anno in self.annotations
            if DataFactory.cast_annotation(anno)
        ]

    def __eq__(self, other):
        if other == self.id:
            return True
        return False


@dataclass_json(undefined=Undefined.INCLUDE)
@dataclass
class AnnotationDefinition:
    id: str
    description: str
    extra_data: CatchAll


@dataclass_json(undefined=Undefined.INCLUDE)
@dataclass
class Annotation:
    type: str = field(metadata=config(field_name="@type"))
    id: str
    sensorId: str
    description: str
    extra_data: CatchAll


class DataFactory:
    """
    Factory class used to register data types that can be deserialized from a solo dataset.
    Annotation, Capture, and Annotation Definition types must register with the annotation factory
    with the following class decorator:
        @DataFactory.register(@type string)
    """

    annotation_switcher = {}
    capture_switcher = {}
    definition_switcher = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """
        Registers a new annotation type.
        Args:
            name (str): Type string for the annotation type
        """

        def wrapper(wrapped) -> Callable:
            if isinstance(wrapped, Annotation) or issubclass(wrapped, Annotation):
                if name in cls.annotation_switcher:
                    logger.warning(f"Annotation {name} has already been registered")
                cls.annotation_switcher[name] = wrapped
                return wrapped
            elif isinstance(wrapped, Capture) or issubclass(wrapped, Capture):
                if name in cls.capture_switcher:
                    logger.warning(f"Capture {name} has already been registered")
                cls.capture_switcher[name] = wrapped
                return wrapped
            elif isinstance(wrapped, AnnotationDefinition) or issubclass(
                wrapped, AnnotationDefinition
            ):
                if name in cls.definition_switcher:
                    logger.warning(
                        f"Annotation definition {name} has already been registered"
                    )
                cls.definition_switcher[name] = wrapped
                return wrapped
            else:
                raise TypeError(
                    "Only can register annotations, captures, or annotation definitions"
                )

        return wrapper

    @classmethod
    def cast_annotation(cls, data):
        if isinstance(data, Annotation):
            return data

        if "@type" not in data.keys():
            raise Exception("No type provided in annotation")

        dtype = data["@type"]
        if dtype not in cls.annotation_switcher.keys():
            logger.info(
                f"Unknown data type: {dtype}. Treating it as generic Annotation type."
            )
            return Annotation.from_dict(data)
        klass = cls.annotation_switcher[dtype]
        return klass.from_dict(data)

    @classmethod
    def cast_capture(cls, data):
        if isinstance(data, Capture):
            return data

        if "@type" not in data.keys():
            raise Exception("No type provided in annotation")

        dtype = data["@type"]
        if dtype not in cls.capture_switcher.keys():
            logger.info(
                f"Unknown data type: {dtype}. Treating it as generic Capture type."
            )
            return Capture.from_dict(data)
        klass = cls.capture_switcher[dtype]
        return klass.from_dict(data)

    @classmethod
    def cast_definition(cls, data):

        if isinstance(data, AnnotationDefinition):
            return data

        if "@type" not in data.keys():
            raise Exception("No type provided in annotation")

        dtype = data["@type"]
        if dtype not in cls.definition_switcher.keys():
            logger.info(
                f"Unknown data type: {dtype}. Treating it as generic AnnotationDefinition type."
            )
            return AnnotationDefinition.from_dict(data)
        klass = cls.definition_switcher[dtype]
        return klass.from_dict(data)


@dataclass
class _BaseMeta:
    metadata: object = field(default_factory=lambda: {}, init=False)


@dataclass
class AnnotationLabel(_BaseMeta):
    instanceId: int
    labelId: int


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
    cameraCartesianLocation: List[float] = field(default_factory=list)


@dataclass
class KeypointLabel(AnnotationLabel):
    pose: str
    keypoints: List[Keypoint]


@dataclass
class InstanceSegmentationLabel(AnnotationLabel):
    labelName: str
    color: List[int]


@dataclass
class SemanticSegmentationLabel(_BaseMeta):
    labelName: str
    pixelValue: List[int]


@DataFactory.register("type.unity.com/unity.solo.KeypointAnnotation")
@dataclass
class KeypointAnnotation(Annotation):
    templateId: str
    values: List[KeypointLabel]


@DataFactory.register("type.unity.com/unity.solo.DepthAnnotation")
@dataclass
class DepthAnnotation(Annotation):
    imageFormat: str
    dimension: List[int]
    filename: str


@DataFactory.register("type.unity.com/unity.solo.PixelPositionAnnotation")
@dataclass
class PixelPositionAnnotation(Annotation):
    imageFormat: str
    dimension: List[int]
    filename: str


@DataFactory.register("type.unity.com/unity.solo.NormalAnnotation")
@dataclass
class NormalAnnotation(Annotation):
    imageFormat: str
    dimension: List[int]
    filename: str


@DataFactory.register("type.unity.com/unity.solo.BoundingBox2DAnnotation")
@dataclass
class BoundingBox2DAnnotation(Annotation):
    values: List[BoundingBox2DLabel] = field(default_factory=list)


@DataFactory.register("type.unity.com/unity.solo.BoundingBox3DAnnotation")
@dataclass
class BoundingBox3DAnnotation(Annotation):
    values: List[BoundingBox3DLabel]


@DataFactory.register("type.unity.com/unity.solo.InstanceSegmentationAnnotation")
@dataclass
class InstanceSegmentationAnnotation(Annotation):
    imageFormat: str
    dimension: List[int]
    filename: str
    instances: List[InstanceSegmentationLabel] = field(default_factory=list)


@DataFactory.register("type.unity.com/unity.solo.SemanticSegmentationAnnotation")
@dataclass
class SemanticSegmentationAnnotation(Annotation):
    imageFormat: str
    dimension: List[int]
    filename: str
    instances: List[SemanticSegmentationLabel]


@dataclass_json
@dataclass
class Frame:
    frame: int
    sequence: int
    step: int
    timestamp: float = 0.0
    metrics: List[dataclass] = field(default_factory=list)
    captures: List[dataclass] = field(default_factory=list)

    def __post_init__(self):
        self.captures = [DataFactory.cast_capture(capture) for capture in self.captures]

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

    def filter_captures(self, sensor):
        res = filter(lambda capture: isinstance(capture, sensor), self.captures)
        return list(res)


@DataFactory.register("type.unity.com/unity.solo.RGBCamera")
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
class KeypointDefinition:
    label: str
    index: int
    color: List[int]


@dataclass
class KeypointTemplateDefinition:
    templateId: str
    templateName: str
    keypoints: List[KeypointDefinition]


@DataFactory.register("type.unity.com/unity.solo.KeypointAnnotation")
@dataclass
class KeypointAnnotationDefinition(AnnotationDefinition):
    template: KeypointTemplateDefinition


@DataFactory.register("type.unity.com/unity.solo.DepthAnnotation")
@dataclass
class DepthAnnotationDefinition(AnnotationDefinition):
    pass  # no additional fields


@DataFactory.register("type.unity.com/unity.solo.PixelPositionAnnotation")
@dataclass
class PixelPositionAnnotationDefinition(AnnotationDefinition):
    pass  # no additional fields


@DataFactory.register("type.unity.com/unity.solo.NormalAnnotation")
@dataclass
class NormalAnnotationDefinition(AnnotationDefinition):
    pass  # no additional fields


@dataclass
class LabelNameSpec:
    label_id: int
    label_name: str


@DataFactory.register("type.unity.com/unity.solo.BoundingBox2DAnnotation")
@dataclass
class BoundingBox2DAnnotationDefinition(AnnotationDefinition):
    spec: List[LabelNameSpec]


@DataFactory.register("type.unity.com/unity.solo.SemanticSegmentationAnnotation")
@dataclass
class SemanticSegmentationAnnotationDefinition(AnnotationDefinition):
    pass  # Adds not additional fields


@DataFactory.register("type.unity.com/unity.solo.BoundingBox3DAnnotation")
@dataclass
class BoundingBox3DAnnotationDefinition(AnnotationDefinition):
    spec: List[LabelNameSpec]


@DataFactory.register("type.unity.com/unity.solo.InstanceSegmentationAnnotation")
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
class DatasetMetadata:
    unityVersion: str
    perceptionVersion: str
    totalFrames: int
    totalSequences: int
    sensors: List[str]
    metricCollectors: List[str]
    annotators: List[object] = field(default_factory=lambda: list())
    scenarioActiveRandomizers: List[str] = field(default_factory=lambda: list())
    simulationStartTime: str = None
    simulationEndTime: str = None
    renderPipeline: str = None
    scenarioRandomSeed: float = None


@dataclass_json
@dataclass
class DatasetAnnotations(object):
    annotationDefinitions: List[dataclass]

    def __post_init__(self):
        self.annotationDefinitions = [
            DataFactory.cast_definition(anno) for anno in self.annotationDefinitions
        ]
