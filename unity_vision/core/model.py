import glob
import json
from abc import ABC, abstractmethod


class RunOutput:
    def __init__(self):
        """"""

class Dataset(ABC):
    """
    Abstract Dataset class
    """
    @abstractmethod
    def get_metadata_file(self):
        pass


class UCVDDataset(Dataset):
    def __init__(
            self,
            root,
            run_id=None,

    ):
        """
        Initializes UCVD Dataset
        """
        self.root = root
        self.run_id = run_id

    def get_metadata_file(self) -> str:
        return f"{self.root}/metadata.json"
