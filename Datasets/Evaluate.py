import sys
import os
import re
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import Trackeval as trackeval  # noqa: E402


def detect_seq_length(file_path):
    """Detect sequence length from a MOT-format file (max frame ID in first column)."""
    if not file_path or not os.path.exists(file_path):
        return None
    max_frame = 0
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            try:
                frame_id = int(float(parts[0]))
                max_frame = max(max_frame, frame_id)
            except (ValueError, IndexError):
                continue
    return max_frame if max_frame > 0 else None


def parse_tracker_filename(filename):
    """Parse Track.py filename structure to extract model name, tracker name, sequence,
    confidence threshold, and video dimensions.

    Expected pattern:
    - Tracker_{model_name}_{tracker_name}_{sequence}_{width}x{height}_conf{conf:.2f}.txt

    Returns:
        tuple: (model_name, tracker_name, sequence, conf_threshold, width, height)
        or (None, None, None, None, None, None) if pattern doesn't match
    """
    if not filename:
        return None, None, None, None, None, None

    stem = Path(filename).stem

    # Check if it matches the Track.py pattern
    if not stem.startswith('Tracker_'):
        return None, None, None, None, None, None

    parts = stem.split('_')
    if len(parts) < 4:
        return None, None, None, None, None, None

    # Extract model name (second part after 'Tracker_')
    model_name = parts[1]

    # Extract tracker name (third part after 'Tracker_')
    tracker_name = parts[2]

    # Check for confidence threshold (last part should be 'conf{value}')
    conf_threshold = None
    if parts[-1].startswith('conf'):
        try:
            conf_threshold = float(parts[-1][4:])
        except ValueError:
            pass

    # Parse dimensions using regex on the full filename stem
    width, height = None, None
    dim_match = re.search(r'(\d+)x(\d+)', stem)
    if dim_match:
        width = int(dim_match.group(1))
        height = int(dim_match.group(2))

    # Check for sequence name (fourth part, if it comes before the dimensions)
    sequence = None
    if len(parts) >= 5 and parts[-1].startswith('conf'):
        # Sequence is parts[3] only if parts[4] is the dimensions (not conf)
        # With new format: Tracker_model_tracker_sequence_1920x1080_conf0.25
        # parts = ['Tracker','model','tracker','sequence','1920x1080','conf0.25']
        # parts[4] should contain 'x' for dimensions
        if 'x' in parts[4] or ('x' in parts[4] and not parts[4].startswith('conf')):
            sequence = parts[3]
        elif 'x' not in parts[4]:
            sequence = parts[3]

    return model_name, tracker_name, sequence, conf_threshold, width, height


def load_mot_data(file_path):
    """Load MOT-format data (6 or 10 columns) from a text file."""
    data = defaultdict(list)
    if not file_path or not os.path.exists(file_path):
        return data
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 6:
                try:
                    frame = int(float(parts[0]))
                    track_id = int(float(parts[1]))
                    x = float(parts[2])
                    y = float(parts[3])
                    w = float(parts[4])
                    h = float(parts[5])
                    data[frame].append({
                        'id': track_id,
                        'x': x,
                        'y': y,
                        'w': w,
                        'h': h
                    })
                except (ValueError, IndexError):
                    continue
    return data


def analyze_crossings_vertical(tracking_data, center_x, total_frames):
    """Count unique tracks that cross a vertical line at center_x."""
    track_positions = defaultdict(list)
    crossing_tracks = set()

    for frame_idx in range(1, total_frames + 1):
        tracks = tracking_data.get(frame_idx, [])
        for track in tracks:
            track_id = track['id']
            x_center = track['x'] + track['w'] / 2
            track_positions[track_id].append((frame_idx, x_center))

    for track_id, positions in track_positions.items():
        if len(positions) < 2:
            continue
        positions.sort()
        for i in range(1, len(positions)):
            prev_x = positions[i - 1][1]
            curr_x = positions[i][1]
            if (prev_x < center_x <= curr_x) or (curr_x < center_x <= prev_x):
                crossing_tracks.add(track_id)
                break

    return len(crossing_tracks)


def analyze_crossings_horizontal(tracking_data, center_y, total_frames):
    """Count unique tracks that cross a horizontal line at center_y."""
    track_positions = defaultdict(list)
    crossing_tracks = set()

    for frame_idx in range(1, total_frames + 1):
        tracks = tracking_data.get(frame_idx, [])
        for track in tracks:
            track_id = track['id']
            y_center = track['y'] + track['h'] / 2
            track_positions[track_id].append((frame_idx, y_center))

    for track_id, positions in track_positions.items():
        if len(positions) < 2:
            continue
        positions.sort()
        for i in range(1, len(positions)):
            prev_y = positions[i - 1][1]
            curr_y = positions[i][1]
            if (prev_y < center_y <= curr_y) or (curr_y < center_y <= prev_y):
                crossing_tracks.add(track_id)
                break

    return len(crossing_tracks)


def calculate_counting_lines(width, height):
    """Calculate default vertical and horizontal counting lines."""
    center_x = width // 2
    center_y = height // 2
    line_spacing_v = width // 4
    line_spacing_h = height // 4

    counting_lines = {
        'left': line_spacing_v,
        'center': center_x,
        'right': width - line_spacing_v
    }
    horizontal_lines = {
        'top': line_spacing_h,
        'middle': center_y,
        'bottom': height - line_spacing_h
    }

    return {**counting_lines, **horizontal_lines}


def calculate_line_crossing_accuracy(tracking_data, gt_data, all_lines, total_frames):
    """Calculate crossing accuracy for all counting lines."""
    results = {}

    for line_name, line_pos in all_lines.items():
        if line_name in ['left', 'center', 'right']:
            crossings = analyze_crossings_vertical(tracking_data, line_pos, total_frames)
            gt_crossings = analyze_crossings_vertical(gt_data, line_pos, total_frames)
        else:
            crossings = analyze_crossings_horizontal(tracking_data, line_pos, total_frames)
            gt_crossings = analyze_crossings_horizontal(gt_data, line_pos, total_frames)

        if gt_crossings > 0:
            if crossings <= gt_crossings:
                crossing_accuracy = (crossings / gt_crossings) * 100
            else:
                crossing_accuracy = (gt_crossings / crossings) * 100
        else:
            crossing_accuracy = 100.0 if crossings == 0 else 0.0

        results[f'Crossings_{line_name}'] = crossings
        results[f'GT_Crossings_{line_name}'] = gt_crossings
        results[f'Crossing_Accuracy_{line_name}'] = crossing_accuracy

    return results


def get_metric_summary_values(metric, results):
    """Extract all computed values from a metric's sequence results.

    Saves every field returned by the metric. For array fields, the mean
    (AUC-style aggregate) is saved as a single scalar value.
    """
    vals = {}

    # Process all fields defined by the metric class
    all_defined_fields = (
        metric.float_fields
        + metric.integer_fields
        + metric.float_array_fields
        + metric.integer_array_fields
    )
    for field in all_defined_fields:
        if field not in results:
            continue
        # Remove metric-specific prefixes from field names
        clean_field = field.replace('CLR_', '').replace('VACE_', '')
        if field in metric.float_array_fields:
            vals[clean_field] = float(100 * np.mean(results[field]))
        elif field in metric.integer_array_fields:
            vals[clean_field] = float(np.mean(results[field]))
        elif field in metric.float_fields:
            vals[clean_field] = float(100 * float(results[field]))
        elif field in metric.integer_fields:
            vals[clean_field] = int(results[field])

    # Include any extra keys returned by eval_sequence but not in the field lists
    for field, value in results.items():
        if field in vals:
            continue
        # Remove metric-specific prefixes from field names
        clean_field = field.replace('CLR_', '').replace('VACE_', '')
        if isinstance(value, np.ndarray):
            vals[clean_field] = float(np.mean(value))
        elif isinstance(value, (int, np.integer)):
            vals[clean_field] = int(value)
        elif isinstance(value, (float, np.floating)):
            vals[clean_field] = float(value)
        else:
            vals[clean_field] = value

    return vals


def save_combined_metrics_csv(output_res, metrics_list, output_csv_path, gt_path=None, tracker_path=None, seq_length=None, include_crossing=False):
    """Save evaluation results to a combined metrics CSV in wide format.

    dataset, tracker, and sequence columns are parsed from the input file paths
    when available, falling back to the evaluator's internal labels otherwise.
    Line crossing accuracy is calculated when 'Crossing' is in the requested
    metrics and video dimensions are parsed from the tracker filename.
    """
    rows = []
    metric_objects = {metric.get_name(): metric for metric in metrics_list}

    # Count is always evaluated by the Evaluator, so add it for CSV generation if present in results
    if 'Count' not in metric_objects and output_res:
        first_dataset = next(iter(output_res.values()))
        if first_dataset:
            first_tracker = next(iter(first_dataset.values()))
            if first_tracker:
                first_seq = next(iter(first_tracker.values()))
                if 'Count' in first_seq:
                    from Trackeval.metrics import Count
                    metric_objects['Count'] = Count()

    # Parse meaningful names from input paths
    def parse_path(path):
        if not path or not os.path.exists(path):
            return None, None, None
        p = Path(path)
        stem = p.stem
        seq = p.parent.name if p.parent else None
        dataset = p.parent.parent.name if p.parent and p.parent.parent else None
        return dataset, seq, stem

    tracker_dataset, tracker_seq, tracker_name = parse_path(tracker_path)
    gt_dataset, gt_seq, _ = parse_path(gt_path)

    # Parse tracker filename to extract additional metadata
    parsed_model_name, parsed_tracker_name, parsed_sequence, conf_threshold, video_width, video_height = parse_tracker_filename(tracker_path)

    # Calculate line crossing accuracy if requested and dimensions are available
    crossing_results = {}
    if (include_crossing and video_width and video_height and gt_path and os.path.exists(gt_path) and
            tracker_path and os.path.exists(tracker_path) and seq_length):
        all_lines = calculate_counting_lines(video_width, video_height)
        gt_data = load_mot_data(gt_path)
        tracking_data = load_mot_data(tracker_path)
        crossing_results = calculate_line_crossing_accuracy(tracking_data, gt_data, all_lines, seq_length)

    for dataset_name, trackers in output_res.items():
        for tracker_name_eval, sequences in trackers.items():
            for seq_name_eval, metric_results in sequences.items():
                # Use parsed names when available, otherwise fall back to evaluator labels
                # Priority: filename parsing > path parsing > evaluator labels
                row = {
                    'dataset': tracker_dataset or gt_dataset or dataset_name,
                    'model': parsed_model_name,
                    'tracker': parsed_tracker_name or tracker_name or tracker_name_eval,
                    'sequence': parsed_sequence or tracker_seq or gt_seq or seq_name_eval,
                }
                # Add confidence threshold if parsed from filename
                if conf_threshold is not None:
                    row['conf_threshold'] = conf_threshold
                # Add line crossing results if available
                if crossing_results:
                    row.update(crossing_results)
                    accuracy_values = [v for k, v in crossing_results.items() if k.startswith('Crossing_Accuracy_')]
                    if accuracy_values:
                        row['Crossing_Accuracy_Mean'] = sum(accuracy_values) / len(accuracy_values)
                for metric_name, metric_obj in metric_objects.items():
                    if metric_name not in metric_results:
                        continue
                    summary = get_metric_summary_values(metric_obj, metric_results[metric_name])
                    for field, value in summary.items():
                        row[field] = value
                rows.append(row)

    df = pd.DataFrame(rows)
    # Capitalise the first letter of each column title
    df.columns = [col[0].upper() + col[1:] if col else col for col in df.columns]

    # Add Resolution column when video dimensions are available
    if video_width and video_height:
        df['Resolution'] = f'{video_width}x{video_height}'

    # Reorder columns so all crossing metrics follow the ML (CLEAR) column,
    # grouped by counting line: left, center, right, top, middle, bottom.
    all_cols = set(df.columns.tolist())
    crossing_prefixes = ('Crossings_', 'GT_Crossings_', 'Crossing_Accuracy_')
    crossing_cols = [c for c in df.columns if c.startswith(crossing_prefixes)]
    non_crossing_cols = [c for c in df.columns if c not in crossing_cols]

    line_order = ['left', 'center', 'right', 'top', 'middle', 'bottom']
    ordered_crossing_cols = []
    for line in line_order:
        for prefix in ('GT_Crossings_', 'Crossings_', 'Crossing_Accuracy_'):
            col_name = f'{prefix}{line}'
            if col_name in all_cols:
                ordered_crossing_cols.append(col_name)
    if 'Crossing_Accuracy_Mean' in all_cols:
        ordered_crossing_cols.append('Crossing_Accuracy_Mean')

    if ordered_crossing_cols:
        # Place crossing metrics after the last CLEAR metric ('Frames'), or just before Identity ('IDF1')
        if 'Frames' in non_crossing_cols:
            insert_idx = non_crossing_cols.index('Frames') + 1
        elif 'IDF1' in non_crossing_cols:
            insert_idx = non_crossing_cols.index('IDF1')
        else:
            insert_idx = len(non_crossing_cols)
        new_order = non_crossing_cols[:insert_idx] + ordered_crossing_cols + non_crossing_cols[insert_idx:]
    else:
        new_order = non_crossing_cols

    # Ensure Resolution is the final column
    if 'Resolution' in new_order:
        new_order = [c for c in new_order if c != 'Resolution'] + ['Resolution']

    df = df[new_order]
    df.to_csv(output_csv_path, index=False)
    print(f"\nCombined metrics CSV saved to: {output_csv_path}")
    return df


if __name__ == '__main__':

    # Command line interface:
    default_dataset_config = {
        'GT_PATH': None,
        'TRACKER_PATH': None,
        'SEQ_LENGTH': None,
    }
    default_metrics_config = {
        'METRICS': ['HOTA', 'CLEAR', 'Identity', 'VACE', 'Crossing'],
        'THRESHOLD': 0.5,
    }

    config = {**default_dataset_config, **default_metrics_config}  # Merge default configs
    parser = argparse.ArgumentParser()
    for setting in config.keys():
        if setting == 'SEQ_LENGTH':
            parser.add_argument(f"--{setting}", type=int)
        elif setting in ['GT_PATH', 'TRACKER_PATH']:
            parser.add_argument(f"--{setting}", type=str)
        elif isinstance(config[setting], list) or config[setting] is None:
            parser.add_argument(f"--{setting}", nargs='+')
        else:
            parser.add_argument(f"--{setting}", type=float)
    args = parser.parse_args().__dict__
    for setting in args.keys():
        if args[setting] is not None:
            config[setting] = args[setting]

    gt_path = config['GT_PATH']
    tracker_path = config['TRACKER_PATH']
    seq_length = config['SEQ_LENGTH']

    # Auto-detect sequence length if not provided
    if seq_length is None:
        detected = detect_seq_length(tracker_path)
        if detected is None:
            detected = detect_seq_length(gt_path)
        if detected is None:
            raise ValueError('SEQ_LENGTH could not be auto-detected. Provide --SEQ_LENGTH or check input files.')
        seq_length = detected
        config['SEQ_LENGTH'] = seq_length
        print(f'Auto-detected SEQ_LENGTH: {seq_length}')

    metrics = config['METRICS']
    threshold = config['THRESHOLD']
    include_crossing = any('Crossing'.lower() == str(m).lower() for m in metrics)

    # Run code
    eval_config = {'PRINT_CONFIG': False}  # Ensure PRINT_CONFIG is in eval_config
    dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
    metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}

    metrics_config['THRESHOLD'] = threshold  # Apply threshold to all metrics

    # Disable HOTA curve plot generation
    trackeval.metrics.HOTA.plot_hota_curve = lambda self, res: None

    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity, trackeval.metrics.VACE]:
        if metric.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric(metrics_config))
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')
    output_res, output_msg = evaluator.evaluate(dataset_list, metrics_list)

    # Save combined metrics to CSV next to the tracker file
    if os.path.isfile(tracker_path):
        csv_dir = os.path.dirname(tracker_path) or '.'
    else:
        csv_dir = tracker_path or '.'
    tracker_name = Path(tracker_path).stem if tracker_path else 'tracker'
    output_csv = os.path.join(csv_dir, f'{tracker_name}_metrics.csv')
    save_combined_metrics_csv(output_res, metrics_list, output_csv, gt_path=gt_path, tracker_path=tracker_path, seq_length=seq_length, include_crossing=include_crossing)
