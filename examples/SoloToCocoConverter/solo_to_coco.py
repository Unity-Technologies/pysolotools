import hashlib
import json
import logging
import os
import shutil
import uuid
from pathlib import Path

import numpy as np
from PIL import Image
from google.protobuf.json_format import MessageToDict
from datetime import datetime
import datasetinsights_local.datasetinsights.constants as const
# from datasetinsights_local.datasetinsights.datasets.unity_perception import (
#     AnnotationDefinitions,
#     Captures,
# )
from datasetinsights_local.datasetinsights.datasets.unity_perception.validation import NoRecordError
from unity_vision.consumers.solo.parser import Solo
from unity_vision.protos.solo_pb2 import (
    BoundingBox2DAnnotation,
    BoundingBox3DAnnotation,
    InstanceSegmentationAnnotation,
    SemanticSegmentationAnnotation,
    KeypointAnnotation
)

logger = logging.getLogger(__name__)


# def uuid_to_int(input_uuid):
#     try:
#         u = uuid.UUID(input_uuid).int
#     except (AttributeError, ValueError):
#         u = int(
#             hashlib.md5(str(input_uuid).encode("utf8")).hexdigest(), base=16
#         )
#
#     return u


class COCOInstancesTransformer():
    """Convert Synthetic dataset to COCO format.
    This transformer convert Synthetic dataset into annotations in instance
    format (e.g. instances_train2017.json, instances_val2017.json)
    Note: We assume "valid images" in the COCO dataset must contain at least one
    bounding box annotation. Therefore, all images that contain no bounding
    boxes will be dropped. Instance segmentation are considered optional
    in the converted dataset as some synthetic dataset might be generated
    without it.
    Args:
        data_root (str): root directory of the dataset
    """

    # The annotation_definition.name is not a reliable way to know the type
    # of annotation definition. This will be improved once the perception
    # package introduced the annotation definition type in the future.
    # BBOX_NAME = r"^(?:2[dD]\s)?bounding\sbox$"
    # INSTANCE_SEGMENTATION_NAME = r"^instance\ssegmentation$"

    # annotation type
    SEMANTIC_SEGMENTATION_TYPE = 'type.unity.com/unity.solo.SemanticSegmentationAnnotationDefinition'
    INSTANCE_SEGMENTATION_TYPE = 'type.unity.com/unity.solo.InstanceSegmentationAnnotationDefinition'
    BOUNDING_BOX_TYPE = 'type.unity.com/unity.solo.BoundingBoxAnnotationDefinition'
    BOUNDING_BOX_3D_TYPE = 'type.unity.com/unity.solo.BoundingBox3DAnnotationDefinition'
    KEYPOINT_TYPE = 'type.unity.com/unity.solo.KeypointAnnotationDefinition'

    def __init__(self, data_root):
        self._data_root = Path(data_root)
        self.get_annotation_definitions()
        self._solo = Solo(data_root)
        self._data_len = self.metadata["totalFrames"]
        self._sensor = self._solo.sensors()[0]['message']
        self._kpts_labeler_exists = True if self._kpt_def is not None else False

    def execute(self, output, **kwargs):
        """Execute COCO Transformer
        Args:
            output (str): the output directory where converted dataset will
              be stored.
        """
        self._copy_images(output)
        self._process_instances(output)

    def _get_annotation_def_by_type(self, annotation: str):
        if annotation == self.SEMANTIC_SEGMENTATION_TYPE:
            return SemanticSegmentationAnnotation()
        if annotation == self.INSTANCE_SEGMENTATION_TYPE:
            return InstanceSegmentationAnnotation()
        if annotation == self.BOUNDING_BOX_TYPE:
            return BoundingBox2DAnnotation()
        if annotation == self.BOUNDING_BOX_3D_TYPE:
            return BoundingBox3DAnnotation()
        if annotation == self.KEYPOINT_TYPE:
            return KeypointAnnotation()
        return None


    def _get_annotation_by_labeler_type(self, annotation, sensor=None):
        sensor = sensor or self._sensor
        annotations = sensor.annotations
        ann_type = self._get_annotation_def_by_type(annotation)

        for a in annotations:
            if a.Is(ann_type.DESCRIPTOR):
                a.Unpack(ann_type)
                messageToDict = MessageToDict(a)
                if 'id' in messageToDict:
                    return messageToDict
        return None

    def get_annotation_definitions(self):
        f = open(os.path.join(self._data_root, "metadata.json"), "r")
        self.metadata = json.load(f)

        f = open(os.path.join(self._data_root, "annotation_definitions.json"), "r")
        self.annotaion_definitions = json.load(f)

        for ann_def in self.annotaion_definitions["annotationDefinitions"]:
            if ann_def["@type"] == self.KEYPOINT_TYPE:
                self._kpt_def = ann_def
            elif ann_def["@type"] == self.BOUNDING_BOX_TYPE:
                self._bbox_def = ann_def

    def _copy_images(self, output):
        image_to_folder = Path(output) / "images"
        image_to_folder.mkdir(parents=True, exist_ok=True)
        start_at = 0
        for index in range(start_at, self._data_len):
            self._solo.__load_frame__(index)
            file_name = f"camera_{index}"
            image_from = os.path.join(self._solo.sequence_path, file_name)
            if not os.path.exists(image_from):
                continue
            image_to = image_to_folder / file_name
            shutil.copy(str(image_from), str(image_to))


    def _process_instances(self, output):
        output = Path(output) / "annotations"
        output.mkdir(parents=True, exist_ok=True)
        date_time = datetime.now()
        date_created = date_time.strftime("%A, %d %B, %Y")
        instances = {
            "info": {
                "year": datetime.now().year,
                "version": "0.0.1",
                "description": "COCO compatible Synthetic Dataset",
                "contributor": "Anonymous",
                "url": "Not Set",
                "date_created": date_created,
            },
            "licences": [
                {
                    "id": 0,
                    "name": "No License",
                    "url": "Not Set"
                }
            ],
            "images": self._images(),
            "annotations": self._annotations(includes_kpt_labeler=self._kpts_labeler_exists),
            "categories": self._categories(),
        }
        output_file = output / "instances.json"
        with open(output_file, "w") as out:
            json.dump(instances, out, indent=3)

    def _images(self):
        images = []
        start_at = 0
        sensor = self._sensor
        for index in range(start_at, self._data_len):
            self._solo.__load_frame__(index)
            image_file = os.path.join(self._solo.sequence_path, sensor.filename)
            if not os.path.exists(image_file):
                continue
            with Image.open(image_file) as im:
                width, height = im.size
            capture_id = index
            date_captured_full = datetime.strptime(self.metadata["simulationStartTime"], "%m/%d/%Y %I:%M:%S %p")
            date_captured = date_captured_full.strftime("%A, %B %d, %Y")
            # file_name = rf"{os.path.relpath(os.path.join(self._solo.sequence_path, sensor.filename), self._solo.path)}"
            file_name = f"camera_{index}.png"
            record = {
                "id": capture_id,
                "width": width,
                "height": height,
                "file_name": file_name,
                "license": 0,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": date_captured,
            }
            images.append(record)

        return images

    def _annotations(self, includes_kpt_labeler: bool):
        if includes_kpt_labeler is True:
            annotations = self._get_coco_record_for_bbx_and_kpts()
        else:
            annotations = self._get_coco_record_for_bbx()
        return annotations

    def _get_coco_record_for_bbx_and_kpts(self):
        annotations = []
        start_at = 0
        for index in range(start_at, self._data_len):
            self._solo.__load_frame__(index)
            bb_dic = self._get_annotation_by_labeler_type(annotation=self.BOUNDING_BOX_TYPE)
            kpt_dic = self._get_annotation_by_labeler_type(annotation=self.KEYPOINT_TYPE)
            image_id = index

            for ann_bb, ann_kpt in zip(bb_dic["values"], kpt_dic["values"]):
                # --- bbox ---
                area = ann_bb["dimension"][0] * ann_bb["dimension"][1]
                # --- keypoint ---
                keypoints_vals = []
                num_keypoints = 0
                for kpt in ann_kpt["keypoints"]:
                    keypoints_vals.append(
                        [
                            kpt["location"][0],
                            kpt["location"][1],
                            kpt["state"] if kpt.get("state") else 0.0
                        ]
                    )
                    if kpt.get("state") and kpt["state"] != 0.0:
                        num_keypoints += 1

                keypoints_vals = [
                    item for sublist in keypoints_vals for item in sublist
                ]

                record = {
                    "id": ann_bb["instanceId"],
                    "image_id": image_id,
                    "category_id": ann_bb["labelId"],
                    "segmentation": [],  # TODO: parse instance segmentation map
                    "area": area,
                    "bbox": ann_bb["origin"] + ann_bb["dimension"],
                    "iscrowd": 0,
                    "num_keypoints": num_keypoints,
                    "keypoints": keypoints_vals,
                }
                annotations.append(record)

        return annotations


    def _get_coco_record_for_bbx(self):
        annotations = []
        start_at = 0
        for index in range(start_at, self._data_len):
            self._solo.__load_frame__(index)
            bb_dic = self._get_annotation_by_labeler_type(annotation=self.BOUNDING_BOX_TYPE)
            image_id = index

            for ann_bb in bb_dic["values"]:
                # --- bbox ---
                area = ann_bb["dimension"][0] * ann_bb["dimension"][1]

                record = {
                    "id": ann_bb["instanceId"],
                    "image_id": image_id,
                    "category_id": ann_bb["labelId"],
                    "segmentation": [],  # TODO: parse instance segmentation map
                    "area": area,
                    "bbox": ann_bb["origin"] + ann_bb["dimension"],
                    "iscrowd": 0,
                }
                annotations.append(record)
        return annotations


    def _categories(self):
        categories = []
        key_points = []
        skeleton = []

        for kp in self._kpt_def["template"]["keypoints"]:
            key_points.append(kp["label"])

        for sk in self._kpt_def["template"]["skeleton"]:
            skeleton.append([sk["joint1"], sk["joint2"]])

        for r in self._bbox_def["spec"]:
            record = {
                "id": r["label_id"],
                "name": r["label_name"],
                "supercategory": "default",
                "keypoints": key_points,
                "skeleton": skeleton,
            }
            categories.append(record)

        return categories

#
# class COCOKeypointsTransformer():
#     """Convert Synthetic dataset to COCO format.
#     This transformer convert Synthetic dataset into annotations
#     in person keypoint format
#     (e.g. person_keypoints_train2017.json, person_keypoints_val2017.json).
#     Note: We assume "valid images" in the COCO dataset must contain at least one
#     bounding box annotation. Therefore, all images that contain no bounding
#     boxes will be dropped. Instance segmentation are considered optional
#     in the converted dataset as some synthetic dataset might be generated
#     without it.
#     Args:
#         data_root (str): root directory of the dataset
#     """
#
#     # The annotation_definition.name is not a reliable way to know the type
#     # of annotation definition. This will be improved once the perception
#     # package introduced the annotation definition type in the future.
#     BBOX_NAME = r"^(?:2[dD]\s)?bounding\sbox$"
#     KPT_NAME = r"^(?:2[dD]\s)?keypoints$"
#     INSTANCE_SEGMENTATION_NAME = r"^instance\ssegmentation$"
#
#     def __init__(self, data_root):
#         self._data_root = Path(data_root)
#
#         ann_def = AnnotationDefinitions(
#             data_root
#         )
#         self._bbox_def = ann_def.find_by_name(self.BBOX_NAME)
#         self._kpt_def = ann_def.find_by_name(self.KPT_NAME)
#         try:
#             self._instance_segmentation_def = ann_def.find_by_name(
#                 self.INSTANCE_SEGMENTATION_NAME
#             )
#         except NoRecordError as e:
#             logger.warning(
#                 "Can't find instance segmentation annotations in the dataset. "
#                 "The converted file will not contain instance segmentation."
#             )
#             logger.warning(e)
#             self._instance_segmentation_def = None
#
#         captures = Captures(
#             data_root=data_root, version=const.DEFAULT_PERCEPTION_VERSION
#         )
#         self._bbox_captures = captures.filter(self._bbox_def["id"])
#         self._kpt_captures = captures.filter(self._kpt_def["id"])
#         if self._instance_segmentation_def:
#             self._instance_segmentation_captures = captures.filter(
#                 self._instance_segmentation_def
#             )
#
#     def execute(self, output, **kwargs):
#         """Execute COCO Transformer
#         Args:
#             output (str): the output directory where converted dataset will
#               be stored.
#         """
#         self._copy_images(output)
#         self._process_instances(output)
#
#     def _copy_images(self, output):
#         image_to_folder = Path(output) / "images"
#         image_to_folder.mkdir(parents=True, exist_ok=True)
#         for _, row in self._bbox_captures.iterrows():
#             image_from = self._data_root / row["filename"]
#             if not image_from.exists():
#                 continue
#             capture_id = uuid_to_int(row["id"])
#             image_to = image_to_folder / f"camera_{capture_id}.png"
#             shutil.copy(str(image_from), str(image_to))
#
#     def _process_instances(self, output):
#         output = Path(output) / "annotations"
#         output.mkdir(parents=True, exist_ok=True)
#         instances = {
#             "info": {"description": "COCO compatible Synthetic Dataset"},
#             "licences": [{"url": "", "id": 1, "name": "default"}],
#             "images": self._images(),
#             "annotations": self._annotations(),
#             "categories": self._categories(),
#         }
#         output_file = output / "keypoints.json"
#         with open(output_file, "w") as out:
#             json.dump(instances, out)
#
#     def _images(self):
#         images = []
#         for _, row in self._bbox_captures.iterrows():
#             image_file = self._data_root / row["filename"]
#             if not image_file.exists():
#                 continue
#             with Image.open(image_file) as im:
#                 width, height = im.size
#             capture_id = uuid_to_int(row["id"])
#             record = {
#                 "file_name": f"camera_{capture_id}.png",
#                 "height": height,
#                 "width": width,
#                 "id": capture_id,
#             }
#             images.append(record)
#
#         return images
#
#     def _annotations(self):
#         annotations = []
#         for [_, row_bb], [_, row_kpt] in zip(
#             self._bbox_captures.iterrows(), self._kpt_captures.iterrows()
#         ):
#             image_id = uuid_to_int(row_bb["id"])
#
#             for ann_bb, ann_kpt in zip(
#                 row_bb["annotation.values"], row_kpt["annotation.values"]
#             ):
#                 # --- bbox ---
#                 x = ann_bb["x"]
#                 y = ann_bb["y"]
#                 w = ann_bb["width"]
#                 h = ann_bb["height"]
#                 area = float(w) * float(h)
#
#                 # -- kpt ---
#                 keypoints_vals = []
#                 num_keypoints = 0
#                 for kpt in ann_kpt["keypoints"]:
#                     keypoints_vals.append(
#                         [
#                             int(np.floor(kpt["x"])),
#                             int(np.floor(kpt["y"])),
#                             kpt["state"],
#                         ]
#                     )
#                     if int(kpt["state"]) != 0:
#                         num_keypoints += 1
#
#                 keypoints_vals = [
#                     item for sublist in keypoints_vals for item in sublist
#                 ]
#
#                 record = {
#                     "segmentation": [],  # TODO: parse instance segmentation map
#                     "area": area,
#                     "iscrowd": 0,
#                     "image_id": image_id,
#                     "bbox": [x, y, w, h],
#                     "keypoints": keypoints_vals,
#                     "num_keypoints": num_keypoints,
#                     "category_id": ann_bb["label_id"],
#                     "id": uuid_to_int(row_bb["annotation.id"])
#                     | uuid_to_int(ann_bb["instance_id"]),
#                 }
#                 annotations.append(record)
#
#         return annotations
#
#     def _categories(self):
#         categories = []
#         key_points = []
#         skeleton = []
#
#         for kp in self._kpt_def["spec"][0]["key_points"]:
#             key_points.append(kp["label"])
#
#         for sk in self._kpt_def["spec"][0]["skeleton"]:
#             skeleton.append([sk["joint1"] + 1, sk["joint2"] + 1])
#
#         for r in self._bbox_def["spec"]:
#             record = {
#                 "id": r["label_id"],
#                 "name": r["label_name"],
#                 "supercategory": "default",
#                 "keypoints": key_points,
#                 "skeleton": skeleton,
#             }
#             categories.append(record)
#
#         return categories
