"""Root package info."""

# import logging
#
# _root_logger = logging.getLogger()
# _logger = logging.getLogger(__name__)
# _logger.setLevel(logging.INFO)
#
# if not _root_logger.hasHandlers():
#     _logger.addHandler(logging.StreamHandler())
#     _logger.propagate = False
#
# from .consumers.solo.parser import SoloBase, Solo
# from .protos.solo_pb2 import (
#     BoundingBox2DAnnotation,
#     BoundingBox3DAnnotation,
#     Frame,
#     InstanceSegmentationAnnotation,
#     RGBCamera)
#
# __all__ = [
#     "SoloBase",
#     "Solo",
#     "BoundingBox2DAnnotation",
#     "BoundingBox3DAnnotation",
#     "Frame",
#     "InstanceSegmentationAnnotation",
#     "RGBCamera",
# ]
#
# __import__("pkg_resources").declare_namespace(__name__)

__all__=[]
