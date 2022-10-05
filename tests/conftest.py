import os
from pathlib import Path

import pytest

from pysolotools.consumers import Solo


@pytest.fixture(scope="session")
def solo_instance() -> Solo:
    input_data_path = os.path.join(Path(__file__).parents[0], "data", "solo")
    return Solo(data_path=input_data_path)


@pytest.fixture(scope="session")
def solo_custom_data_instance() -> Solo:
    input_data_path = os.path.join(
        Path(__file__).parents[0], "data", "solo_custom_type"
    )
    return Solo(data_path=input_data_path)
