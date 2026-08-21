# Deploy

This directory contains a range of deployment architectures integrating **ByteTraX**. These consist of models supported by a custom implementation of the [Ultralytics](https://github.com/ultralytics/ultralytics) package, covering tasks including [detection](#yolo-detection), [segmentation](#yolo-segmentation), [pose-estimation](#yolo-pose-estimation), and [OBB](#yolo-obb).

<div align="center"><img src="../Images/Boat%20OBB%20Example.gif" width="32%" alt="Boat OBB Example"> <img src="../Images/Shark%20Breach%20Segmentation%20Example.gif" width="32%" alt="Shark Breach Segmentation Example"> <img src="../Images/Dance%20Pose-estimation%20Example.gif" width="32%" alt="Dance Pose-estimation Example"></div>

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

## Tasks

<a id="yolo-detection"></a>
YOLO detection.

```bash
yolo track model=yolo26n.pt source="path/to/video.mp4" tracker="bytetrax.yaml"
```

<a id="yolo-segmentation"></a>
YOLO Segmentation.

```bash
yolo segment model=yolo26n-seg.pt source="path/to/video.mp4" tracker="bytetrax.yaml"
```

<a id="yolo-pose-estimation"></a>
YOLO Pose-estimation.

```bash
yolo pose model=yolo26n-pose.pt source="path/to/video.mp4" tracker="bytetrax.yaml"
```

<a id="yolo-obb"></a>
YOLO OBB.

```bash
yolo obb model=yolo26n-obb.pt source="path/to/video.mp4" tracker="bytetrax.yaml"
```

## Export

Export models in a variety of formats for edge inference including [LiterRT](https://docs.ultralytics.com/integrations/litert), [ONNX](https://docs.ultralytics.com/integrations/onnx), and [NCNN](https://docs.ultralytics.com/integrations/ncnn).

```bash
yolo export model=yolo26n.pt format=onnx
```

For a full list of supported formats, see the Ultralytics [export mode](https://github.com/ultralytics/ultralytics/blob/main/docs/en/modes/export.md) documentation.

## Contribute

Contributions that integrate additional model frameworks are welcomed.
