import os
import pathlib
import shutil

from unity_vision.consumers.solo.parser import SoloBase
from unity_vision.protos.solo_pb2 import (BoundingBox2DAnnotation,
                                          BoundingBox3DAnnotation,
                                          InstanceSegmentationAnnotation,
                                          KeypointAnnotation, RGBCamera,
                                          SemanticSegmentationAnnotation)
from SoloToCocoConverter.solo_to_coco import COCOInstancesTransformer
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

import unittest
from os import listdir
from os.path import isfile, join, exists
from pycocotools.coco import COCO
import os

output_path = "temp_output/"
annotation_output = os.path.join(output_path, "annotations")
image_output = os.path.join(output_path, "images")
annFile = os.path.join(annotation_output, "instances.json")
data_len = 10

class TestBasic(unittest.TestCase):
    def setUp(self):
        # Load test data
        self.app = COCOInstancesTransformer(data_root='data/solo')
        self.app.execute(output=output_path)

    def test_output_file_count(self):
        output_image_files = [f for f in listdir(image_output) if isfile(join(image_output, f))]
        self.assertEqual(len(output_image_files), data_len)
        self.assertTrue(exists(annFile))
        # clean_up()

    def test_output_with_coco(self):
        # initialize COCO api for instance annotations
        coco = COCO(annFile)
        # display COCO categories and supercategories
        cats = coco.loadCats(coco.getCatIds())
        nms = [cat['name'] for cat in cats]
        self.assertEqual(nms, ['Crate', 'Cube', 'Box', 'Terrain', 'Character'])
        print('COCO categories: \n{}\n'.format(' '.join(nms)))

        nms = set([cat['supercategory'] for cat in cats])
        self.assertEqual(nms, {'default'})
        print('COCO supercategories: \n{}'.format(' '.join(nms)))

    def test_output_image_info(self):
        coco = COCO(annFile)
        for index in range(0, data_len):
            img = coco.loadImgs(index)[0]
            self.assertEqual(img['id'], index)
            self.assertEqual(img['width'], 822)
            self.assertEqual(img['height'], 509)
            self.assertEqual(img['file_name'], f'camera_{index}.png')

    def test_output_contains_bbox(self):
        # initialize COCO api for person keypoints annotations
        coco = COCO(annFile)
        coco_kps = COCO(annFile)
        for index in range(0, data_len):
            img = coco.loadImgs(index)[0]
            annIds = coco_kps.getAnnIds(img['id'], iscrowd=None)
            anns = coco_kps.loadAnns(annIds)
            self.assertEqual(anns[0]['image_id'], index)
            self.assertTrue(anns[0]['bbox'], True)

    # delete test output after finished testing
    def tearDown(self):
        try:
            shutil.rmtree(output_path)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

if __name__ == '__main__':
    unittest.main()


