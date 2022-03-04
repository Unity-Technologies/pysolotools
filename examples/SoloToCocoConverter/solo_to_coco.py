import json
import logging
import os
import shutil
from pathlib import Path
from google.protobuf.json_format import MessageToDict
from datetime import datetime
from unity_vision.consumers.solo.parser import Solo
from unity_vision.protos.solo_pb2 import (
    BoundingBox2DAnnotation,
    BoundingBox3DAnnotation,
    InstanceSegmentationAnnotation,
    SemanticSegmentationAnnotation,
    KeypointAnnotation
)

logger = logging.getLogger(__name__)


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
        self._data_len = self._metadata["totalFrames"]
        self._sensor = self._solo.sensors()[0]['message']
        self._sensor_name = "unity.solo.RGBCamera"
        self._kpts_labeler_exists = True if hasattr(self, "_kpt_def") else False
        self._images = []
        self._annotations = []
        self._categories = []

    def execute(self, output, **kwargs):
        """Execute COCO Transformer
        Args:
            output (str): the output directory where converted dataset will
              be stored.
        """
        solo = self._solo

        # --- process each frame at once
        for idx, current_frame in enumerate(solo):
            if idx == self._data_len:
                break
            self._images.append(self._process_image(current_frame, idx, output))
            self._annotations.append(self._process_annotation(idx, includes_kpt_labeler=self._kpts_labeler_exists))
            self._categories = self._process_categories(includes_kpt_labeler=self._kpts_labeler_exists)

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
            "images": self._images,
            "annotations": self._annotations,
            "categories": self._categories
        }
        ann_output = Path(output) / "annotations"
        ann_output.mkdir(parents=True, exist_ok=True)
        output_file = ann_output / "instances.json"
        with open(output_file, "w") as out:
            json.dump(instances, out, indent=3)

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
        self._metadata = json.load(f)

        f = open(os.path.join(self._data_root, "annotation_definitions.json"), "r")
        self._annotaion_definitions = json.load(f)

        for ann_def in self._annotaion_definitions["annotationDefinitions"]:
            if ann_def["@type"] == self.KEYPOINT_TYPE:
                self._kpt_def = ann_def
            elif ann_def["@type"] == self.BOUNDING_BOX_TYPE:
                self._bbox_def = ann_def

    def _process_image(self, current_frame, idx, output):
        sensor = self._sensor_name
        image_to_folder = Path(output) / "images"
        image_to_folder.mkdir(parents=True, exist_ok=True)
        current_frame_info = current_frame[sensor]
        image_file = os.path.join(self._solo.sequence_path, current_frame_info['filename'])

        # --- copy images to coco output folder ---
        sensor_id = current_frame_info['id']
        image_to_file = f"{sensor_id}_{idx}.png"
        image_to = image_to_folder / image_to_file
        shutil.copy(str(image_file), str(image_to))

        width, height = current_frame_info['dimension'][0], current_frame_info['dimension'][1]
        date_captured_full = datetime.strptime(self._metadata["simulationStartTime"], "%m/%d/%Y %I:%M:%S %p")
        date_captured = date_captured_full.strftime("%A, %B %d, %Y")
        record = {
            "id": idx,
            "width": width,
            "height": height,
            "file_name": image_to_file,
            "license": 0,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": date_captured,
        }
        return record

    def _process_annotation(self, idx, includes_kpt_labeler):
        if includes_kpt_labeler is True:
            annotations = self._get_coco_record_for_bbx_and_kpts(idx)
        else:
            annotations = self._get_coco_record_for_bbx(idx)
        return annotations

    def _get_coco_record_for_bbx_and_kpts(self, idx):
        bb_dic = self._get_annotation_by_labeler_type(annotation=self.BOUNDING_BOX_TYPE)
        kpt_dic = self._get_annotation_by_labeler_type(annotation=self.KEYPOINT_TYPE)

        for ann_bb, ann_kpt in zip(bb_dic["values"], kpt_dic["values"]):
            record = self._get_coco_record_for_bbx(idx)
            # --- bbox ---
            # area = ann_bb["dimension"][0] * ann_bb["dimension"][1]

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

            record['num_keypoints'] = num_keypoints
            record['keypoints'] = keypoints_vals

            # record = {
            #     "id": idx,
            #     "image_id": idx,
            #     "category_id": ann_bb["labelId"],
            #     "segmentation": [],  # TODO: parse instance segmentation map
            #     "area": area,
            #     "bbox": ann_bb["origin"] + ann_bb["dimension"],
            #     "iscrowd": 0,
            #     "num_keypoints": num_keypoints,
            #     "keypoints": keypoints_vals,
            # }
        return record

    def _get_coco_record_for_bbx(self, idx):
        bb_dic = self._get_annotation_by_labeler_type(annotation=self.BOUNDING_BOX_TYPE)

        for ann_bb in bb_dic["values"]:
            # --- bbox ---
            area = ann_bb["dimension"][0] * ann_bb["dimension"][1]

            record = {
                "id": idx,
                "image_id": idx,
                "category_id": ann_bb["labelId"],
                "segmentation": [],  # TODO: parse instance segmentation map
                "area": area,
                "bbox": ann_bb["origin"] + ann_bb["dimension"],
                "iscrowd": 0,
            }
        return record

    def _process_categories(self, includes_kpt_labeler=bool):
        categories = []
        if includes_kpt_labeler is True:
            key_points = []
            skeleton = []
            for kp in self._kpt_def["template"]["keypoints"]:
                key_points.append(kp["label"])

            for sk in self._kpt_def["template"]["skeleton"]:
                skeleton.append([sk["joint1"] + 1, sk["joint2"] + 1])

            for r in self._bbox_def["spec"]:
                record = self._bbox_categories(r)
                record['keypoints'] = key_points
                record['skeleton'] = skeleton
                # record = {
                #     "id": r["label_id"],
                #     "name": r["label_name"],
                #     "supercategory": "default",
                #     "keypoints": key_points,
                #     "skeleton": skeleton,
                # }
                categories.append(record)
        else:
            for r in self._bbox_def["spec"]:
                # record = {
                #     "id": r["label_id"],
                #     "name": r["label_name"],
                #     "supercategory": "default",
                # }
                record = self._bbox_categories(r)
                categories.append(record)
        return categories

    def _bbox_categories(self, r):
        # for r in self._bbox_def["spec"]:
        record = {
            "id": r["label_id"],
            "name": r["label_name"],
            "supercategory": "default",
        }
        return record


if __name__ == "__main__":
    dataset = COCOInstancesTransformer(data_root=os.path.join(Path(__file__).parents[1], "data", "solo"))
    dataset.execute(output=os.path.join(Path(__file__).parents[1], "data_output"))
