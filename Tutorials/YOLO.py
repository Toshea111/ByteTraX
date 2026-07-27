"""Tutorial: track objects in a video using the GMOT-40 YOLO model + the ByteTraX tracker.

Requires the `bytetrax` package to be installed:
    py -m pip install -e Deploy/Ultralytics

Usage:
    py Tutorials/YOLO.py
"""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

VIDEO_PATH = Path(__file__).resolve().parent / "Videos" / "GMOT-40.mp4"
MODEL_PATH = Path(__file__).resolve().parents[1] / "Models" / "YOLO" / "GMOT-40Model.pt"
OUTPUT_DIR = Path(__file__).resolve().parent
RUNS_DIR = OUTPUT_DIR / "YOLO Runs"
OUTPUT_PATH = RUNS_DIR / f"{VIDEO_PATH.stem}.avi"


def main() -> None:
    model = YOLO(str(MODEL_PATH))
    results = model.track(
        source=str(VIDEO_PATH),
        tracker="bytetrax.yaml",
        save=True,
        save_dir=str(RUNS_DIR),
        conf=0.25,
    )
    print(f"Processed {len(results)} frames. Tracked output saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
