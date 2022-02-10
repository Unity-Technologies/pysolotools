import glob

from torch.utils.data import IterableDataset

from unity_vision.consumers.solo.parser import Solo


class SoloDataset(IterableDataset):
    def __init__(self, path):
        """
        Flattens a Solo dataset with 1 frame element per sequence. Refer to solo.proto.
        Return an iterator that starts from sequence_idx and iterates through all sequences
        returning the Frame from each of them.

        :param path: data root containing sequences
        """
        self.base_path = path
        self.sequences = glob.glob(f"{self.base_path}/sequence.*")
        self.sensor = "unity.solo.RGBCamera"
        # TODO: Support allowing custom annotation protobufs
        self.solo = Solo(self.base_path, sensors=[self.sensor], start=0, end=self.__len__())

    def __len__(self):
        return len(self.sequences)

    def __iter__(self):
        return self

    def __next__(self):
        d = next(self.solo)[self.sensor]
        return d["filename"], d["annotations"]
