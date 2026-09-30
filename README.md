# ByteTraX

**ByteTraX** is a multi-object tracking algorithm developed as an enhancement of the [ByteTrack](https://github.com/FoundationVision/ByteTrack) architecture. It leverages optimised thresholding paired with track reconnection and merging functions to reduce identity switches and improve tracking continuity. This yields superior speed and accuracy across a range of tracking tasks and benchmarks.

<div align="center"><img src="Images/GMOT-40%20Example.gif" width="32%" alt="GMOT-40 Example"> <img src="Images/Shark%20Breach%20Example.gif" width="32%" alt="Shark Breach Example"> <img src="Images/TeamTrack%20Example.gif" width="32%" alt="TeamTrack Example"></div>

## Setup

Clone the repository and create a conda environment.

```bash
git clone https://github.com/Toshea111/ByteTraX.git
cd ByteTraX
conda create -n bytetrax python=3.12 -y
```

## Installation

Activate the environment and install the required packages.

```bash
conda activate bytetrax
pip install -e Deploy/ultralytics
```

## Usage

### Python

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
results = model.track(source="path/to/video.mp4", tracker="bytetrax.yaml")
```

### CLI

```bash
yolo track model=yolo26n.pt source="path/to/video.mp4" tracker="bytetrax.yaml"
```

## Configuration

ByteTraX features a track reconnection and merging functionality that may be activated for improved performance in closed systems—those with fixed track counts. This can be toggled along with additional parameters via the [bytetrax.yaml](Deploy/ultralytics/cfg/trackers/bytetrax.yaml) configuration file.

## Demo

Run the tutorial script for a full working example.

```bash
python Tutorials/YOLO.py
```

## Deploy

Rapidly deploy ByteTraX using a range of model architectures across [detection](Deploy#yolo-detection), [segmentation](Deploy#yolo-segmentation), and [pose-estimation](Deploy#yolo-pose-estimation) tasks, and [export](Deploy#export) compatible models in a variety of formats for edge inference. See [Deploy](Deploy) for currently supported models, tasks, and formats, along with training instructions via the Ultralytics platform.

## Evaluate

Evaluate ByteTraX on your own models and benchmarks using the [Track](#Track) and [Evaluate](#Evaluate) scripts in [Datasets](Datasets).

## Pretrained Models

Pretrained YOLO26n model weights for the benchmarks used in testing can be downloaded from [Models](Models).

## Contribute

Contributions and suggestions to improve the ByteTraX algorithm or integrate additional model architectures are welcomed.

## Acknowledgements

ByteTraX is built upon the [ByteTrack](https://github.com/ifzhang/ByteTrack) architecture, and utilises the [Ultralytics](https://github.com/ultralytics/ultralytics) framework for object detection.

## Citation

If you find ByteTraX useful, please consider citing the publication.

```bibtex
@article{osheawheller2026bytetraxenhancingbytetrackarchitecture,
  title={ByteTraX: Enhancing the ByteTrack Architecture with Optimised Thresholding}, 
  author={Thomas A. O'Shea-Wheller},
  year={2026},
  eprint={2609.37801},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2609.37801}, 
}
```
