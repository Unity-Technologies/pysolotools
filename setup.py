#!/usr/bin/env python
"""
Installs pysolotools
"""

import io
import json
import os
from os.path import dirname, realpath

from setuptools import find_packages, setup

# Package meta-data.
NAME = "pysolotools"
DESCRIPTION = "unity computer vision dataset toolchain"
URL = "https://https://github.com/Unity-Technologies/pysolotools"
EMAIL = "computer-vision@unity3d.com"
AUTHOR = "Unity Technologies"
REQUIRES_PYTHON = ">=3.7"
FALL_BACK_VERSION = "0.3.16"


here = os.path.abspath(os.path.dirname(__file__))

try:
    with io.open(
        os.path.join(here, "github_release_version.json"), encoding="utf-8"
    ) as f:
        VERSION = json.loads(f.read()).get("version", FALL_BACK_VERSION)
except FileNotFoundError:
    VERSION = FALL_BACK_VERSION


def _read_requirements():
    requirements = f"{dirname(realpath(__file__))}/requirements/base.txt"
    with open(requirements) as f:
        results = []
        for line in f:
            line = line.strip()
            if "-i" not in line:
                results.append(line)
        return results


setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    py_modules=[NAME],
    include_package_data=True,
    license="MIT",
    install_requires=_read_requirements(),
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    packages=find_packages(include=["pysolotools", "pysolotools.*"]),
    entry_points={
        "console_scripts": [
            "solo2yolo=pysolotools.converters.solo2yolo:cli",
            "solo2coco=pysolotools.converters.solo2coco:cli",
        ]
    },
)
