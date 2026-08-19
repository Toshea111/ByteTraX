"""
Script to run tracking algorithms with YOLO26 models on videos and save outputs in MOTChallenge format.

Usage examples:
    # Single video
    python Track.py --video <path/to/video.mp4> --model <path/to/model.pt> --tracker <path/to/tracker.yaml> --conf 0.25 --output <output_dir>

    # Dataset directory with multiple video folders
    python Track.py --base_dir <path/to/dataset> --folders <folder1> <folder2> --model <path/to/model.pt> --tracker <path/to/tracker.yaml> --conf 0.25 --output <output_dir>
"""

import argparse
import os
import sys
import cv2
from pathlib import Path
from ultralytics import YOLO


def find_video(base_dir, folder):
    """Find video file (.mov or .mp4) inside a dataset folder."""
    folder_path = os.path.join(base_dir, folder)
    if not os.path.exists(folder_path):
        return None

    # Look for any .mp4 or .mov file in the folder
    for ext in [".mov", ".mp4"]:
        # First try the folder name pattern
        video_path = os.path.join(folder_path, f"{folder}{ext}")
        if os.path.exists(video_path):
            return video_path

    # If not found, look for any video file in the folder
    for file in os.listdir(folder_path):
        if file.endswith((".mov", ".mp4")):
            return os.path.join(folder_path, file)

    return None


def process_video(model, video_path, tracker_config, conf_threshold, output_path):
    """Run tracking on a single video and save results in MOT format."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"  Video: {video_path}")
    print(f"  Total frames: {total_frames}")
    print(f"  Output: {output_path}")

    results = model.track(
        source=video_path,
        tracker=tracker_config,
        conf=conf_threshold,
        stream=True,
        verbose=False,
    )

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        frame_id = 0
        for result in results:
            frame_id += 1
            if frame_id % 50 == 0 and total_frames > 0:
                progress = (frame_id / total_frames) * 100
                print(f"  Processing frame {frame_id}/{total_frames} ({progress:.1f}%)...")
            if result.boxes is not None:
                for box in result.boxes:
                    if box.id is not None:
                        track_id = int(box.id.item())
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        bbox_left = x1
                        bbox_top = y1
                        bbox_width = x2 - x1
                        bbox_height = y2 - y1
                        f.write(
                            f"{frame_id},{track_id},{bbox_left:.2f},{bbox_top:.2f},"
                            f"{bbox_width:.2f},{bbox_height:.2f}\n"
                        )

    print(f"  Completed. Output saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run YOLO + tracker on videos and save MOT-format tracking results."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the YOLO model (.pt file).",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        required=True,
        help="Path to the tracker config YAML file (e.g., bytetrax.yaml, bytetrack.yaml).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections (default: 0.25).",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to a single video file to process.",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=None,
        help="Base directory containing dataset video folders.",
    )
    parser.add_argument(
        "--folders",
        type=str,
        nargs="+",
        default=None,
        help="List of video folder names inside --base_dir to process.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results"),
        help="Output directory for MOT-format tracking results (default: script_dir/Results).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional output filename prefix. Default is Tracker_<tracker_name>_<folder>_conf<conf>.",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.video and not args.base_dir:
        parser.error("Either --video or --base_dir must be provided.")
    if args.base_dir and not args.folders:
        parser.error("--folders is required when using --base_dir.")
    if args.video and args.base_dir:
        parser.error("Provide either --video or --base_dir, not both.")

    tracker_name = Path(args.tracker).stem
    model_name = Path(args.model).stem

    print(f"Loading YOLO model: {args.model}")
    model = YOLO(args.model)

    if args.video:
        # Single video mode
        cap = cv2.VideoCapture(args.video)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        video_name = Path(args.video).stem
        output_filename = args.name or f"Tracker_{model_name}_{tracker_name}_{video_name}_{width}x{height}_conf{args.conf:.2f}.txt"
        output_path = os.path.join(args.output, output_filename)
        print(f"\nProcessing single video with tracker: {tracker_name}, conf={args.conf}")
        process_video(model, args.video, args.tracker, args.conf, output_path)
    else:
        # Dataset folder mode
        print(f"\nUsing tracker: {tracker_name}")
        for folder in args.folders:
            video_path = find_video(args.base_dir, folder)
            if not video_path:
                print(f"Warning: Video not found for folder {folder}")
                continue

            # Use actual video filename as sequence
            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            video_name = Path(video_path).stem
            output_filename = args.name or f"Tracker_{model_name}_{tracker_name}_{video_name}_{width}x{height}_conf{args.conf:.2f}.txt"
            output_path = os.path.join(args.output, output_filename)

            print(f"\nProcessing {video_name}...")
            process_video(model, video_path, args.tracker, args.conf, output_path)

    print("\nAll videos processed successfully!")


if __name__ == "__main__":
    main()
