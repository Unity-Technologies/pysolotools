<div align="center">
<p align="center">


**The open-source tool for loading and analyzing Unity SOLO datasets**
---

[![PyPI version](https://github.com/pytest-dev/pytest-cov/actions/workflows/test.yml/badge.svg)](https://github.com/Unity-Technologies/pysolotools/actions)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

</p>
</div>

## Introduction

pysolotools is a python package for managing SOLO datasets.
It helps to perform following tasks:

- Parse SOLO datasets generated with Unity Perception
- Dataset iterables
- Convert SOLO to [COCO format](https://cocodataset.org/#format-data)
- Compute common statistics

You can read more about SOLO schema [here](https://github.com/Unity-Technologies/perception/blob/main/com.unity.perception/com.unity.perception/Documentation~/SoloSchema/Solo_Schema.md).

## Pre-Requisites
- Install [Anaconda](https://docs.anaconda.com/anaconda/install/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (recommended). Install [pre-commit](https://pre-commit.com/).
- Make sure `pip` is installed.

## Installation

```shell
pip install pysolotools --index-url=https://artifactory.prd.it.unity3d.com/artifactory/api/pypi/pypi/simple
```

#### SOLO Dataset


##### Load and iterate frames

```python
from pysolotools.consumers import Solo
solo = Solo(data_path="src_data_path")

for frame in solo.frames():
    # perform operations on frame
```

##### SOLO2COCO conversion
Supports conversion for these labels: 2d bbox, keypoints, instance, semantic.

```python
from pysolotools.converters.solo2coco import SOLO2COCOConverter
from pysolotools.consumers import Solo

solo = Solo("src_data_path")
dataset = SOLO2COCOConverter(solo)
dataset.convert(output_path="output_path")
```
##### Stats computation
Supports bbox, keypoints and image analysis on SOLO dataset.

```python
from pysolotools.consumers import Solo
from pysolotools.stats.analyzers.bbox_analyzer import BBoxHeatMapStatsAnalyzer, BBoxSizeStatsAnalyzer
from pysolotools.stats.handler import StatsHandler

bbheat=BBoxHeatMapStatsAnalyzer()
bbsize=BBoxSizeStatsAnalyzer()
solo = Solo("data_path")
bbh= StatsHandler(solo=solo)
bbh.handle(analyzers=[bbheat,bbsize],cat_ids=[])
```


## Community and Feedback

The Unity Computer Vision demos are open-source and we encourage and welcome contributions.
If you wish to contribute, be sure to review our [contribution guidelines](CONTRIBUTING.md)
and [code of conduct](CODE_OF_CONDUCT.md).

## Support

For feature requests, bugs, or other issues, please file a
[GitHub issue](https://github.com/Unity-Technologies/Unity-Vision-Hub/issues)
using the provided templates we will investigate as soon as possible.


## License
[Apache License 2.0](LICENSE)
