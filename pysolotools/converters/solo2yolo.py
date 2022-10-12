import argparse
import logging
import multiprocessing
import shutil
import sys
from pathlib import Path

from pysolotools.consumers import Solo
from pysolotools.core import BoundingBox2DAnnotation, RGBCameraCapture
from pysolotools.core.models import Frame

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class Solo2YoloConverter:
    def __init__(self, solo: Solo):
        self._solo = solo
        self._pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    @staticmethod
    def _filter_annotation(ann_type, annotations):
        filtered_ann = list(
            filter(
                lambda k: isinstance(
                    k,
                    ann_type,
                ),
                annotations,
            )
        )
        if filtered_ann:
            return filtered_ann[0]
        return filtered_ann

    @staticmethod
    def _process_rgb_image(image_id, rgb_capture, output, data_root, sequence_num):
        image_file = data_root / f"sequence.{sequence_num}/{rgb_capture.filename}"
        image_to_file = f"camera_{image_id}.png"
        image_to = output / image_to_file
        shutil.copy(str(image_file), str(image_to))

    @staticmethod
    def _to_yolo_bbox(img_width, img_height, center_x, center_y, box_width, box_height):
        x = (center_x + box_width * 0.5) / img_width
        y = (center_y + box_height * 0.5) / img_height
        w = box_width / img_width
        h = box_height / img_height

        return x, y, w, h

    @staticmethod
    def _process_annotations(image_id, rgb_capture, output):
        width, height = rgb_capture.dimension
        print(f"w: {width}, h: {height}")
        filename = f"camera_{image_id}.txt"
        file_to = output / filename

        bbox_ann = Solo2YoloConverter._filter_annotation(
            ann_type=BoundingBox2DAnnotation, annotations=rgb_capture.annotations
        ).values

        with open(str(file_to), "w") as f:
            for bbox in bbox_ann:
                x, y, w, h = Solo2YoloConverter._to_yolo_bbox(
                    width,
                    height,
                    bbox.origin[0],
                    bbox.origin[1],
                    bbox.dimension[0],
                    bbox.dimension[1],
                )
                f.write(f"{bbox.labelId} {x} {y} {w} {h}\n")

    @staticmethod
    def _process_instances(frame: Frame, idx, images_output, labels_output, data_root):
        print(f"Processing frame number: {idx}", flush=True)
        image_id = idx
        sequence_num = frame.sequence

        # Currently support only single camera
        rgb_capture = list(
            filter(lambda cap: isinstance(cap, RGBCameraCapture), frame.captures)
        )[0]

        Solo2YoloConverter._process_rgb_image(
            image_id, rgb_capture, images_output, data_root, sequence_num
        )
        Solo2YoloConverter._process_annotations(image_id, rgb_capture, labels_output)

    @staticmethod
    def _print_this(idx):
        print("Hi", flush=True)
        return idx * 3

    def convert(self, output_path: str, dataset_name: str = "yolo"):
        print(f"in convert")
        base_path = Path(output_path) / dataset_name
        images_output = base_path / "images"
        labels_output = base_path / "labels"
        images_output.mkdir(parents=True, exist_ok=True)
        labels_output.mkdir(parents=True, exist_ok=True)

        data_path = Path(self._solo.data_path)

        for idx, frame in enumerate(self._solo.frames()):
            print(f"processing frame {idx}")

            # res = self._pool.apply_async(self._print_this, (idx,))
            # print(f"returned frame {res.get()}")

            self._pool.apply_async(
                self._process_instances,
                args=(frame, idx, images_output, labels_output, data_path),
            )

        self._pool.close()
        self._pool.join()


def cli():
    parser = argparse.ArgumentParser(
        prog="solo2yolo",
        description=("Converts SOLO datasets into YOLO datasets",),
        epilog="\n",
    )

    parser.add_argument("solo")

    args = parser.parse_args(sys.argv[1:])

    solo = Solo(args.solo)

    converter = Solo2YoloConverter(solo)

    converter.convert("D:/PerceptionOutput/Tutorial")


if __name__ == "__main__":
    cli()
