from unity_vision.consumers.solo.parser import Solo


def run():
    solo = Solo(path="data/solo/", start=0, end=10)
    # Sample 10 datapoints
    while True:
        try:
            frame = next(solo)
            annotations, metrics = [cap.annotations for cap in frame.captures], frame.metrics
            print(annotations)
            print(metrics)
        except StopIteration:
            break


if __name__ == "__main__":
    run()
