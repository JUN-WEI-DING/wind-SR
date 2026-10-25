import os
from pathlib import Path

import pytest

ENV = {"data": "WINDSR_DATA"}


def pytest_collection_modifyitems(items):
    for item in items:
        for mark, var in ENV.items():
            if item.get_closest_marker(mark) and not os.environ.get(var):
                item.add_marker(pytest.mark.skip(reason=f"{var} is not set"))


@pytest.fixture(scope="session")
def data() -> Path:
    """The data directory of a `data` test."""
    return Path(os.environ["WINDSR_DATA"])
