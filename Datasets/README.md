# Datasets

This directory contains scripts for saving tracking results in MOTChallenge format, and evaluating them via the [TrackEval Lite](https://github.com/30-A/trackeval_lite) framework. This enables the generation of standard MOT metrics concordant with [TrackEval](https://github.com/JonathonLuiten/TrackEval), but without the need for specific directory structures or sequence information files. It can be used to evaluate **ByteTraX** and other tracking algorithms on a range of benchmarks.

## Track

Run a tracking algorithm with a YOLO26 model and save the results in MOTChallenge format.

### Usage

For a single video.

```bash
python Track.py --video <path/to/video.mp4> --model <path/to/model.pt> --tracker <path/to/tracker.yaml> --conf 0.25
```
For all videos within a directory.

```bash
python Track.py --base_dir <path/to/dataset> --folders <folder1> <folder2> --model <path/to/model.pt> --tracker <path/to/tracker.yaml> --conf 0.25
```

### Arguments

- `--model`: Path to the YOLO26 model.
- `--tracker`: Path to the tracker config YAML file.
- `--conf`: Confidence threshold for detections (Default 0.25).
- `--video`: Path to a single video file for processing.
- `--base_dir`: Base directory containing multiple video folders
- `--folders`: List of video folder names inside `--base_dir`.
- `--output`: Output directory for results (Default `script_dir/Results`).
- `--name`: Optional output file name prefix.

### Output

This generates MOTChallenge results files with the following format.

```
<frame_id>,<track_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>
```

## Evaluate

Evaluate tracking results against ground truth data using the TrackEval framework.

### Usage

```bash
python Evaluate.py --GT_PATH <path/to/gt.txt> --TRACKER_PATH <path/to/tracker_output.txt>
```

### Arguments

- `--GT_PATH`: Path to ground truth data file.
- `--TRACKER_PATH`: Path to tracker results file.
- `--METRICS`: Metrics to compute (Defualt HOTA, CLEAR, Identity, VACE).
- `--THRESHOLD`: Ground truth IoU threshold for evaluation (Default 0.5).

### Output

This generates a combined `.csv` file with the specified evaluation metrics. Results are summarised by dataset (parsed from `--base_dir`), video sequence (parsed from `--video` or `--folders`), tracker (parsed from `--tracker`), model (parsed from `--model`), and confidence threshold (parsed from `--conf`) for ease of analysis.

## Workflow

The evaluation workflow consists of a few simple steps, with the following providing a working example.

### Setup

Ensure that benchmark videos are in `.mp4` or `.mov` format, and that the corresponding ground truth files use the [six column](#output) MOTChallenge format. The [ten column](https://github.com/JonathonLuiten/TrackEval/tree/master/docs/MOTChallenge-Official) format used for 3D benchmarks is also supported. No specific folder structure is required, however it can be useful to place files within a base directory named after the benchmark.

### Generate Results

Use `Track.py` to generate tracking results by specifying the model, video, and tracker locations.

```bash
python Track.py --video "Tutorials/Videos/SportsMOT.mp4" --model "Models/YOLO/SportsMOTModel.pt" --tracker "Deploy/ultralytics/cfg/trackers/bytetrax.yaml"
```

### Evaluate Results

Use `Evaluate.py` to evaluate the results by specifying the ground truth and tracker results file locations. Run variables and sequence lengths will be extracted automatically from the file paths.

```bash
python Evaluate.py --GT_PATH "Data/SportsMOTGT.txt" --TRACKER_PATH "Results/Tracker_SportsMOTModel_bytetrax_SportsMOT_conf0.25.txt"
```

## Modifications

In practice, it may be useful to modify `Track.py` and `Evaluate.py` to automatically cycle through models, trackers, and confidence thresholds as required by your testing regime. While these scripts currently require manual path specification, they are readily adaptable to such enhancements.

## Acknowledgements

This pipeline was developed using the [TrackEval Lite](https://github.com/30-A/trackeval_lite) and [TrackEval](https://github.com/JonathonLuiten/TrackEval) frameworks.
