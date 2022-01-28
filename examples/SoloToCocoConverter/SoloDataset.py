import json
import os
from os import listdir
from os.path import isfile, join
from typing import Dict
#from PIL import Image
from google.protobuf.json_format import MessageToDict
from unity_vision.consumers.solo.parser import Solo
from unity_vision.protos.solo_pb2 import (
    BoundingBox2DAnnotation,
    BoundingBox3DAnnotation,
    InstanceSegmentationAnnotation,
    SemanticSegmentationAnnotation,
    KeypointAnnotation
)
#import visualization.visualizers as v

SEMANTIC_SEGMENTATION_TYPE = 'type.unity.com/unity.solo.SemanticSegmentationAnnotationDefinition'
INSTANCE_SEGMENTATION_TYPE = 'type.unity.com/unity.solo.InstanceSegmentationAnnotationDefinition'
BOUNDING_BOX_TYPE = 'type.unity.com/unity.solo.BoundingBoxAnnotationDefinition'
BOUNDING_BOX_3D_TYPE = 'type.unity.com/unity.solo.BoundingBox3DAnnotationDefinition'
KEYPOINT_TYPE = 'type.unity.com/unity.solo.KeypointAnnotationDefinition'


class Dataset:
    @staticmethod
    def check_folder_valid(base_dataset_dir: str):
        found_solo_meta = False
        found_solo_annotate = False
        found_solo_metric = False
        found_solo_sensor = False
        try:
            meta_file = "metadata.json"
            annotate_file = "annotation_definitions.json"
            metric_file = "metric_definitions.json"
            sensor_file = "sensor_definitions.json"
            files = [f for f in listdir(base_dataset_dir) if isfile(join(base_dataset_dir, f))]
            if meta_file in files:
                found_solo_meta = True
            if annotate_file in files:
                found_solo_annotate = True
            if metric_file in files:
                found_solo_metric = True
            if sensor_file in files:
                found_solo_sensor = True
            return found_solo_meta and found_solo_annotate and found_solo_metric and found_solo_sensor
        except PermissionError:
            return False

    def __init__(self, data_root: str):
        if Dataset.check_folder_valid(data_root):
            try:
                self.data_root = data_root
                self.get_annotation_definitions()
                self.solo = Solo(data_root, start=0)
                self.dataset_valid = True
            except Exception as e:
                print(e)
                self.data_root = None
                self.solo = None
                self.dataset_valid = False
        else:
            self.data_root = None
            self.solo = None

            self.dataset_valid = False

    def get_annotation_definitions(self):
        f = open(os.path.join(self.data_root, "metadata" + "." + "json"), "r")
        self.metadata = json.load(f)

        f = open(os.path.join(self.data_root, "annotation_definitions.json"), "r")
        self.annotaion_definitions = json.load(f)
        return self.metadata

    def get_available_labelers(self):
        return self.metadata["annotators"]

    def length(self):
        return self.metadata["totalFrames"]

    def get_keypoint_template(self, templateId: str):
        for i in self.annotaion_definitions['annotationDefinitions']:
            if i['@type'] == KEYPOINT_TYPE:
                template = i['template']
                if template['templateId'] == templateId:
                    return template;

        return None

    def _to_annotation(self, annotation):
        if annotation == SEMANTIC_SEGMENTATION_TYPE:
            return SemanticSegmentationAnnotation()
        if annotation == INSTANCE_SEGMENTATION_TYPE:
            return InstanceSegmentationAnnotation()
        if annotation == BOUNDING_BOX_TYPE:
            return BoundingBox2DAnnotation()
        if annotation == BOUNDING_BOX_3D_TYPE:
            return BoundingBox3DAnnotation()
        if annotation == KEYPOINT_TYPE:
            return KeypointAnnotation()
        return None

    # def get_annotator_dictionary(self):
    #     available_labelers = self.get_available_labelers()
    #     annotator_dic = {}
    #     for labeler in available_labelers:
    #         annotator_name_state = self.AnnotatorNameState(labeler['name'], False)
    #         if labeler['type'] not in annotator_dic:
    #             annotator_dic[labeler['type']] = [annotator_name_state]
    #         else:
    #             annotator_dic[labeler['type']].append(annotator_name_state)
    #     return annotator_dic

    def _get_annotation_from_sensor(self, sensor, annotator, annotation):
        annotations = sensor.annotations
        ann_type = self._to_annotation(annotation)

        for a in annotations:
            if a.Is(ann_type.DESCRIPTOR):
                a.Unpack(ann_type)
                messageToDict = MessageToDict(a)
                if 'id' in messageToDict and messageToDict['id'] == annotator.name and annotator.state:
                    return messageToDict
                elif 'id' not in messageToDict and annotator.state:
                    return messageToDict

        return None


    # create AnnotatorNameState class
    # class AnnotatorNameState:
    #     def __init__(self, name: str, state: bool):
    #         self.name = name
    #         self.state = state



    # def get_solo_image_with_labelers(
    #         self,
    #         index: int,
    #         labelers_to_use: Dict[str, bool],
    #         annotator_dic: Dict[str, AnnotatorNameState],
    #         max_size: int = 500) -> Image:
    #
    #     self.solo.jump_to(index)
    #
    #     sensor = self.solo.sensors()[0]['message']
    #     part_a = self.solo.sequence_path
    #     part_b = sensor.filename
    #     filename = os.path.join(self.solo.sequence_path, sensor.filename)
    #     image = Image.open(filename)
    #
    #     if labelers_to_use is None:
    #         labelers_to_use = []
    #
    #     if BOUNDING_BOX_TYPE in labelers_to_use and labelers_to_use[BOUNDING_BOX_TYPE]:
    #         for annotator in annotator_dic[BOUNDING_BOX_TYPE]:
    #             if annotator.state:
    #                 bbox_data = self._get_annotation_from_sensor(sensor, annotator,
    #                                                              BOUNDING_BOX_TYPE)
    #                 label_mappings = {
    #                     m["labelId"]: m["labelName"] for m in bbox_data['values']
    #                 }
    #                 image = v.draw_solo_image_with_boxes(image, bbox_data, label_mappings)
    #
    #     if KEYPOINT_TYPE in labelers_to_use and labelers_to_use[KEYPOINT_TYPE]:
    #         for annotator in annotator_dic[KEYPOINT_TYPE]:
    #             if annotator.state:
    #                 keypoint_data = self._get_annotation_from_sensor(sensor, annotator,
    #                                                                  KEYPOINT_TYPE)
    #                 b = keypoint_data['templateId']
    #                 template = self.get_keypoint_template(b)
    #                 image = v.draw_image_with_keypoints(image, keypoint_data, template)
    #
    #     if BOUNDING_BOX_3D_TYPE in labelers_to_use and labelers_to_use[BOUNDING_BOX_3D_TYPE]:
    #         for annotator in annotator_dic[BOUNDING_BOX_3D_TYPE]:
    #             if annotator.state:
    #                 bbox_3d_data = self._get_annotation_from_sensor(sensor, annotator,
    #                                                                 BOUNDING_BOX_3D_TYPE)
    #                 image = v.draw_image_with_box_3d(image, sensor, bbox_3d_data, None)
    #
    #     if SEMANTIC_SEGMENTATION_TYPE in labelers_to_use and labelers_to_use[SEMANTIC_SEGMENTATION_TYPE]:
    #         for annotator in annotator_dic[SEMANTIC_SEGMENTATION_TYPE]:
    #             if annotator.state:
    #                 seg_data = self._get_annotation_from_sensor(sensor, annotator,
    #                                                             SEMANTIC_SEGMENTATION_TYPE)
    #                 seg_filename = os.path.join(self.solo.sequence_path, seg_data['filename'])
    #                 seg = Image.open(seg_filename)
    #                 image = v.draw_image_with_segmentation(
    #                     image, seg
    #                 )
    #
    #     if INSTANCE_SEGMENTATION_TYPE in labelers_to_use and labelers_to_use[INSTANCE_SEGMENTATION_TYPE]:
    #         for annotator in annotator_dic[INSTANCE_SEGMENTATION_TYPE]:
    #             if annotator.state:
    #                 inst_data = self._get_annotation_from_sensor(sensor, annotator,
    #                                                             INSTANCE_SEGMENTATION_TYPE)
    #                 inst_filename = os.path.join(self.solo.sequence_path, inst_data['filename'])
    #                 inst = Image.open(inst_filename)
    #                 image = v.draw_image_with_segmentation(
    #                     image, inst
    #                 )
    #
    #     return image
