import os
from pathlib import Path

from solo_to_coco import COCOInstancesTransformer as Solo_COCOInstancesTransformer

def run_solo():
    dataset = Solo_COCOInstancesTransformer(data_root=os.path.join(Path(__file__).parents[1], "data","solo"))
    output = dataset.execute(output=os.path.join(Path(__file__).parents[1], "data_output"))
    print(output)

if __name__ == "__main__":
    run_solo()