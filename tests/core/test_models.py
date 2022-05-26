from pysolo.core.models import Frame


def test_frame_rgb_path():
    expected_path = "sequence.0/step0.camera.png"
    f_path = "tests/data/solo/sequence.0/step0.frame_data.json"
    with open(f_path, "r") as f:
        frame = Frame.from_json(f.read())
        rgb_img_path = frame.get_rgb_image_path()
        assert rgb_img_path == expected_path
