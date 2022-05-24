from pysolo.consumers import Solo


def run():
    solo = Solo(path="data/solo/", start=0, end=10)
    print("Dataset Metadata !! ")
    print(solo.get_metadata())
    while True:
        try:
            frame = next(solo)
            annotations, metrics = [
                cap.annotations for cap in frame.captures
            ], frame.metrics
            print(annotations)
            print(metrics)
        except StopIteration:
            break


if __name__ == "__main__":
    run()
