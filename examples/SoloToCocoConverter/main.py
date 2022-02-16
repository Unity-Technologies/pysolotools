from solo_dataset import SoloDataset
from legacy_to_coco import COCOInstancesTransformer, COCOKeypointsTransformer
def run():
    dataset = SoloDataset(path="data/solo")
    # Sample 10 datapoints
    for i in range(1):
        filename, annotations = dataset.__next__()
        print(annotations[0])
        print(f"Filename: {filename}\nAnnotations: {annotations}")

def run_legacy():
    dataset = COCOKeypointsTransformer(data_root=r"C:\Users\ruiyu.zhang\Desktop\legacyToCoco\legacy")
    output = dataset.execute(output=r"C:\Users\ruiyu.zhang\Desktop\legacyToCoco\output2")
    print(output)

def run_solo():
    dataset = COCOKeypointsTransformer(data_root=r"C:\Users\ruiyu.zhang\Desktop\legacyToCoco\solo")
    output = dataset.execute(output=r"C:\Users\ruiyu.zhang\Desktop\legacyToCoco\output_solo")
    print(output)

if __name__ == "__main__":
    # run()
    # run_legacy()
    run_solo()