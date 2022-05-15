from solo_dataset import SoloDataset


def run():
    dataset = SoloDataset(path="data/", annotation_file="data/metadata.json")
    # Sample 10 datapoints
    for _, frame in enumerate(dataset):
        return ([cap.annotations for cap in frame.captures], frame.metrics)
        # print(frame captures)


if __name__ == "__main__":
    annotations, metrics = run()

    print(f"Frame annotations: {annotations}")
    print(f"Frame metrics: {metrics}")
