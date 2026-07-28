# Models

This directory contains pretrained detection models for each of the six benchmarks utilised in the evaluation of **ByteTraX**. These can be used for testing of additional tracking algorithms.

## YOLO

Pretrained YOLO26n detection models for tracking benchmarks. Each model can be paired with the corresponding video in `Tutorials/Videos/` for a demonstration.

## Benchmarks

Full details of the corresponding benchmarks are provided as follows.

- **DAMUNT** — [Code](https://github.com/chamathabeysinghe/da-tracker), [publication](https://arxiv.org/pdf/2301.10559)
- **DeepSea-MOT** — [Code](https://github.com/mbari-org/benchmark_eval), [publication](https://arxiv.org/abs/2509.03499)
- **GMOT-40** — [Code](https://github.com/Spritea/GMOT40), [publication](https://arxiv.org/abs/2011.11858)
- **LC-MOT** — [Code](https://doi.org/10.5281/zenodo.17719647), [publication](https://link.springer.com/article/10.1007/s00371-026-04456-4)
- **SportsMOT** — [Code](https://github.com/MCG-NJU/SportsMOT), [publication](https://arxiv.org/abs/2304.05170)
- **TeamTrack** — [Code](https://github.com/AtomScott/TeamTrack), [publication](https://arxiv.org/abs/2404.13868)

## Usage

Load a model and track the corresponding video.

```python
from ultralytics import YOLO

model = YOLO("Models/YOLO/GMOT-40Model.pt")
results = model.track(source="Tutorials/Videos/GMOT-40.mp4", tracker="bytetrax.yaml")
```
