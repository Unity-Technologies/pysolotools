import os
import tempfile

import pytest

from pysolotools.clients.file_strategy import (
    FileStrategy,
    LocalFileStrategy,
    NoOpFileStrategy,
)


class TestFileStrategy:
    def test_read(self):
        with pytest.raises(TypeError) as _:
            FileStrategy().read()

    def test_write(self):
        with pytest.raises(TypeError) as _:
            FileStrategy().write(contents="foo")


class TestNoOpFileStrategy:
    def test_read(self):
        assert NoOpFileStrategy().read() is None

    def test_write(self):
        assert NoOpFileStrategy().write(contents="foo") is None


class TestLocalFileStrategy:
    filename = "somefile.txt"
    contents = "I am stuff"

    def test_read(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            subject = LocalFileStrategy(
                destination_directory=tmp_dir, filename=self.filename
            )
            subject.write(contents=self.contents)
            assert subject.read() == self.contents

    def test_write(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            LocalFileStrategy(
                destination_directory=tmp_dir, filename=self.filename
            ).write(contents=self.contents)

            with open(os.path.join(tmp_dir, self.filename), "r") as the_file:
                result = the_file.read()
                assert result == self.contents
