# ByteTraX

**ByteTraX** is a multi-object tracking algorithm developed as an enhancement of the [ByteTrack](https://github.com/FoundationVision/ByteTrack) architecture. It leverages optimised thresholding paired with track reconnection and merging functions to reduce identity switches and improve tracking continuity. This yields superior speed and accuracy across a range of tracking tasks and benchmarks. It is paired with [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for detection.

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

Deploy via Python.

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
results = model.track(source="path/to/video.mp4", tracker="bytetrax.yaml")
```

## CLI

Deploy via the Ultralytics CLI.

```bash
yolo track model=yolo26n.pt source="path/to/video.mp4" tracker="bytetrax.yaml"
```

## Demo

Run the tutorial script for a full working example.

```bash
python Tutorials/YOLO.py
```
