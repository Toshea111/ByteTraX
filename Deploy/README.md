# Deploy

This directory contains a range of deployment architectures integrating **ByteTraX**. These consist of models supported by a custom implementation of the [Ultralytics](https://github.com/ultralytics/ultralytics) package, covering tasks including detection, segmentation, pose-estimation, and prompting.

<div align="center"><img src="../Images/GMOT-40%20Example%202.gif" width="32%" alt="GMOT-40 Example 2"> <img src="../Images/Shark%20Breach%20Segmentation%20Example.gif" width="32%" alt="Shark Breach Segmentation Example"> <img src="../Images/TeamTrack%20Segmentation%20Example.gif" width="32%" alt="TeamTrack Segmentation Example"></div>

## Architectures

- **YOLO** — Standard `detect`, `segment`, `pose`, and `obb`
- **RT-DETR** — Transformer-based `detect`
- **YOLO-NAS** — Neural architecture search for `detect`
- **YOLOWorld** — Open-vocabulary for `detect`
- **YOLOE** — Visual and text prompting for `detect` and `segment`

## Installation

Activate the environment and install the required packages.

```bash
conda activate bytetrax
pip install -e Deploy/ultralytics
```

## Usage

### Python

```python
from ultralytics import YOLO, RTDETR, NAS, YOLOWorld, YOLOE
```
YOLO example.

```
model = YOLO("yolo26n.pt")
results = model.track(source="path/to/video.mp4", tracker="bytetrax.yaml")
```

RT-DETR example.

```
model = RTDETR("rtdetr-l.pt")
results = model.track(source="path/to/video.mp4", tracker="bytetrax.yaml")
```

### CLI

YOLO example.

```bash
yolo track model=yolo26n.pt source="path/to/video.mp4" tracker="bytetrax.yaml"
```

RT-DETR example.

```bash
yolo track model=rtdetr-l.pt source="path/to/video.mp4" tracker="bytetrax.yaml"
```
