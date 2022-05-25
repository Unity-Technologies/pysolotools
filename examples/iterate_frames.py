from pysolo.consumers import Solo


def run():
    solo = Solo(data_path="data/solo/", start=0, end=10)
    print("Dataset Metadata !! ")
    print(solo.get_metadata())
    stats = solo.stats
    print(stats.get_frame_ids())

    for frame in solo.frames():
        annotations, metrics = [
            cap.annotations for cap in frame.captures
        ], frame.metrics
        print(annotations)
        print(metrics)


if __name__ == "__main__":
    run()
