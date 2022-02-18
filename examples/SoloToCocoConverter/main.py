from solo_to_coco import COCOInstancesTransformer as Solo_COCOInstancesTransformer

def run_solo():
    dataset = Solo_COCOInstancesTransformer(data_root=r"C:\Users\ruiyu.zhang\Desktop\legacyToCoco\solo_1kpt")
    output = dataset.execute(output=r"C:\Users\ruiyu.zhang\Desktop\legacyToCoco\output_solo_1kpt")
    print(output)

if __name__ == "__main__":
    run_solo()