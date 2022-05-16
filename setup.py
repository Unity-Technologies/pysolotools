#!/usr/bin/env python
"""
Installs unity_vision
"""

import io
import os

from setuptools import setup

# Package meta-data.
NAME = "unity_vision"
DESCRIPTION = "unity computer vision toolchain"
URL = "https://https://github.com/Unity-Technologies/unity-vision"
EMAIL = "souranil@unity3d.com"
AUTHOR = "Unity Technologies"
REQUIRES_PYTHON = ">=3.6"
VERSION = "0.2.2"


here = os.path.abspath(os.path.dirname(__file__))

try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, "__version__.py")) as f:
        exec(f.read(), about)
else:
    about["__version__"] = VERSION

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    py_modules=[NAME],
    include_package_data=True,
    license="MIT",
    install_requires=[
        "protobuf>=3.17.2"
    ],
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    packages=["unity_vision", "unity_vision.consumers.solo", "unity_vision.protos"]
)
