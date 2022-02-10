from solo_dataset import SoloDataset


def run():
    dataset = SoloDataset(path="data/solo")
    # Sample 10 datapoints
    for i in range(10):
        filename, annotations = dataset.__next__()
        print(f"Filename: {filename}\nAnnotations: {annotations}")


if __name__ == "__main__":
    run()
