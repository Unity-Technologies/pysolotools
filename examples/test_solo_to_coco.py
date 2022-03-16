import os
import shutil
import unittest
from os import listdir
from os.path import exists, isfile, join

from pycocotools.coco import COCO
from SoloToCocoConverter.solo_to_coco import COCOInstancesTransformer

from unity_vision.protos.solo_pb2 import (BoundingBox2DAnnotation,
                                          BoundingBox3DAnnotation,
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

    def test_images_produced(self):
        output_image_files = [f for f in listdir(image_output) if isfile(join(image_output, f))]
        self.assertEqual(data_len, len(output_image_files))

    def test_annotation_file_exists(self):
        self.assertTrue(exists(annFile))

    def test_output_categories_with_coco(self):
        # initialize COCO api for instance annotations
        coco = COCO(annFile)
        # display COCO categories and supercategories
        cats = coco.loadCats(coco.getCatIds())
        nms = [cat['name'] for cat in cats]
        self.assertEqual(['Crate', 'Cube', 'Box', 'Terrain', 'Character'], nms)
        print('COCO categories: \n{}\n'.format(' '.join(nms)))

        nms = set([cat['supercategory'] for cat in cats])
        self.assertEqual({'default'}, nms)
        print('COCO supercategories: \n{}'.format(' '.join(nms)))

    def test_output_image_info(self):
        coco = COCO(annFile)
        for index in range(0, data_len):
            img = coco.loadImgs(index)[0]
            self.assertEqual(index, img['id'])
            self.assertEqual(822, img['width'])
            self.assertEqual(509, img['height'])
            self.assertEqual(f'camera_{index}.png', img['file_name'])

    def test_output_contains_bbox(self):
        # initialize COCO api for person keypoints annotations
        coco = COCO(annFile)
        coco_kps = COCO(annFile)
        for index in range(0, data_len):
            img = coco.loadImgs(index)[0]
            annIds = coco_kps.getAnnIds(img['id'], iscrowd=None)
            anns = coco_kps.loadAnns(annIds)
            coco_kps.showAnns(anns, draw_bbox=True)
            self.assertEqual(index, anns[0]['image_id'])
            self.assertTrue(True, anns[0]['bbox'])

    def test_annotations_correct_number(self):
        coco = COCO(annFile)
        total_num_anns = len(coco.dataset['annotations'])
        # 3 bounding boxes and 1 keypoint labelers for each frame
        self.assertEqual(data_len * 3, total_num_anns)

    # delete test output after finished testing
    def tearDown(self):
        try:
            shutil.rmtree(output_path)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))


if __name__ == '__main__':
    unittest.main()
