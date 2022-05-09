#!/usr/bin/env python

"""
Provides a base parser to import solo datasets.

Solo:
    Returns an iterator to read datasets which have 1 frame per sequence. This essentially flattens
    all the sequences and gives an iterator to loop through all the frames in the dataset along with
     the corresponding sensors and their annotations.

SoloSequence:
    Returns iterator to read all steps in a sequence.

"""


import glob
import json
import time
from abc import ABC, abstractmethod
from typing import Iterable

from google.protobuf.json_format import MessageToDict, Parse

from unity_vision.protos.solo_pb2 import (BoundingBox2DAnnotation,
                                          BoundingBox3DAnnotation, Frame,
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


class SoloBase(ABC):
    def __init__(self):
        self.sensor_pool = self._init_sensor_pool()

    @staticmethod
    def _init_sensor_pool():
        sensor_pool = dict()
        for sensor in __SENSORS__:
            s_message = sensor["sensor"]
            sensor_pool[s_message.DESCRIPTOR.full_name] = {
                "message": s_message(),
                "annotations": dict(
                    (annotation_msg.DESCRIPTOR.full_name, annotation_msg())
                    for annotation_msg in sensor["annotations"]
                ),
            }
        return sensor_pool

    def _unpack_annotations(self, annotations, pool):
        unpacked_annotations = list()
        for annotation_msg in annotations:
            a_type = annotation_msg.TypeName()
            # Known annotations
            if a_type in pool.keys():
                annotation_u_msg = pool[annotation_msg.TypeName()]
                annotation_msg.Unpack(annotation_u_msg)
                anno = MessageToDict(annotation_u_msg)
                anno["type"] = a_type
                unpacked_annotations.append(anno)
            else:
                # TODO: For unknown structures return a plain dict with a custom type.
                raise Exception(
                    f"Annotation {annotation_msg.DESCRIPTOR.full_name} doesn't belong to sensor"
                )
        return unpacked_annotations

    def _unpack_sensors(self, sensor_captures):
        sensors = dict()
        for sensor_msg in sensor_captures:
            s_type = sensor_msg.TypeName()
            if s_type in self.sensor_pool.keys():
                sensor_datum = self.sensor_pool[sensor_msg.TypeName()]
                sensor_u_msg = sensor_datum["message"]
                sensor_msg.Unpack(sensor_u_msg)
                sensor_u_annotations = self._unpack_annotations(
                    sensor_u_msg.annotations, sensor_datum["annotations"]
                )
                sensor_data = MessageToDict(sensor_u_msg)
                sensor_data["type"] = s_type
                sensor_data["annotations"] = sensor_u_annotations
                sensors[s_type] = sensor_data
            else:
                raise Exception("Failed to unpack sensor. Unknown sensor")
        return sensors

    def parse_frame(self, f_path):
        f = open(f_path, "r")
        f_message = json.dumps(json.load(f))
        frame = Parse(
            f_message, Frame(), ignore_unknown_fields=True, descriptor_pool=None
        )
        sensors = self._unpack_sensors(frame.captures)
        for s in sensors.values():
            s['frame'] = frame.frame
            s['sequence'] = frame.sequence
            s['step'] = frame.step

        return sensors

    def sensors(self):
        return list(self.sensor_pool.values())

    @abstractmethod
    def register_annotation_to_sensor(self, sensor, annotation):
        raise NotImplementedError

    @abstractmethod
    def __iter__(self) -> Iterable:
        raise NotImplementedError

    @abstractmethod
    def __next__(self):
        raise NotImplementedError


class Solo(SoloBase):
    """
    Parser for solo dataset with 1 frame per sequence. Returns a list of frames from each sequence.
    Essentially flattens the sequence.
    """

    def __init__(self,
                 path,
                 annotation_file=None,
                 sensors=None,
                 start=0,
                 end=None,
                 *args,
                 **kwargs):
        """
        Constructor of Unity SOLO class for reading annotations.
        Args:
            path (str): Path to dataset. This should have all sequences.
            annotation_file (str): Location of annotation file
            sensors (list[dict]): A list of sensor objects to be read from the dataset.
            start (int): Start sequence
            end (int): End sequence
            metadata_path (str): Optional kwarg for providing custom metadata json file path.
        """
        super().__init__()
        self.frame_pool = list()
        self.path = path
        self.frame_idx = start

        """
        Default metadata location is expected at root/metadata.json but
        if an annotation_file path is provided that is used as the annotation

        Metadata can be in one of two locations, depending if it was a part of a singular build,
        or if it was a part of a distributed build.
        """
        metadata_f = self.__open_metadata__(annotation_file)
        pre = time.time()
        metadata = json.load(metadata_f)
        print('DONE (t={:0.5f}s)'.format(time.time() - pre))

        self.total_frames = metadata["totalFrames"]
        self.total_sequences = metadata["totalSequences"]
        self.steps_per_sequence = (int)(self.total_frames / self.total_sequences)

        self.end = end or self.__len__()

        # sensor filter
        if sensors:
            self.sensor_pool = {
                k: self.sensor_pool[k] for k in sensors if k in self.sensor_pool.keys()
            }

    def __open_metadata__(self, annotation_file=None):
        discovered_path = None
        if annotation_file:
            discovered_path = annotation_file
        else:
            discovered_path = glob.glob(self.path + "/**/metadata.json", recursive=True)
            if len(discovered_path) != 1:
                raise Exception("Found multiple metadata files.")
        return open(discovered_path[0])

    def register_annotation_to_sensor(self, sensor, annotation):
        if not sensor.DESCRIPTOR:
            raise Exception(
                "Invalid sensor message. Please pass new sensor protobuf message descriptor"
            )
        if sensor.DESCRIPTOR.full_name not in self.sensor_pool.keys():
            raise Exception("Sensor doesn't exist to add annotations to.")
        sensor_data = self.sensor_pool[sensor.DESCRIPTOR.full_name]
        if annotation.DESCRIPTOR.full_name not in sensor_data["annotations"].keys():
            sensor_data["annotations"][annotation.DESCRIPTOR.full_name] = annotation()

    def __load_frame__(self, frame):
        self.frame_idx = frame
        sequence = (int)(frame / self.steps_per_sequence)
        step = frame % self.steps_per_sequence
        self.sequence_path = f"{self.path}/*sequence.{sequence}"
        filename_pattern = f"{self.sequence_path}/step{step}.frame_data.json"
        files = glob.glob(filename_pattern)
        # There should be exactly 1 frame_data for a particular sequence.
        if len(files) != 1:
            raise Exception(f"Metadata file not found for sequence {sequence}")
        self.frame_idx += 1
        return self.parse_frame(files[0])

    def __iter__(self):
        return self

    def __next__(self):
        if self.frame_idx >= self.end:
            raise StopIteration
        return self.__load_frame__(self.frame_idx)

    def __len__(self):
        return self.total_frames


class SoloSequence(SoloBase):
    def __init__(self, path, sensors=None, start=0, end=None):
        super().__init__()
        self.frame_pool = list()
        self.sequence_path = path
        self.step_idx = start  # Tracks the frame in a sequence
        # TODO: Should be this from metadata ?
        self.sequence_length = len(glob.glob(f"{self.sequence_path}/*.json"))
        if not end:
            self.end = self.sequence_length - 1
        else:
            self.end = end

    def register_annotation_to_sensor(self, sensor, annotation):
        pass

    def __iter__(self) -> Iterable:
        return self

    def __next__(self):
        if self.step_idx >= self.end:
            raise StopIteration

        sequence_data = list()
        for frame_path in glob.glob(f"{self.sequence_path}/*.json"):
            sensors = self.parse_frame(frame_path)
            data = dict()
            for s in sensors.values():
                image_path = s["fileName"]
                annotations = s["annotations"]
                data[s["type"]] = (image_path, annotations)
            self.step_idx += 1
            sequence_data.append(data)
        return sequence_data
