import os
from abc import ABC, abstractmethod
from pathlib import Path


class FileStrategy(ABC):
    """
    Abstract class defining the file strategy
    """

    @abstractmethod
    def write(self, contents: str) -> None:
        """
        Returns None.
        Args:
            contents (str): the contents to write

        Returns:
            None

        """
        pass

    @abstractmethod
    def read(self) -> str:
        """
        Returns string of the contents of a file.
        Args:

        Returns:
            The file as a string

        """
        pass


class LocalFileStrategy(FileStrategy):
    directory: str
    filename: str

    def __init__(self, destination_directory: str, filename: str):
        self.directory = destination_directory
        self.filename = filename

    def read(self) -> str:
        with open(os.path.join(self.directory, self.filename), "r") as file:
            return file.read()

    def write(self, contents: str) -> None:
        dest_dir = Path(self.directory)
        dest_dir.mkdir(parents=True, exist_ok=True)
        output_file = os.path.join(dest_dir, f"{self.filename}")
        print("writing data to file", output_file)
        with open(output_file, "w") as out:
            out.write(contents)


class NoOpFileStrategy(FileStrategy):
    def read(self) -> str:
        pass

    def write(self, contents: str) -> None:
        pass
