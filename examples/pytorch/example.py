from solo_dataset import SoloDataset


def run():
    dataset = SoloDataset(path="data/solo/")
    # Sample 10 datapoints
    for _ in range(10):
        frame = next(dataset)
        return ([cap.annotations for cap in frame.captures], frame.metrics)
        # print(frame captures)


if __name__ == "__main__":
    annotations, metrics = run()

    print(f"Frame annotations: \n{annotations}")
    print(f"Frame metrics: \n{metrics}")
