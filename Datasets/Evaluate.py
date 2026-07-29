import sys
import os
import argparse
from pathlib import Path
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
        if field in metric.float_array_fields:
            vals[field] = float(100 * np.mean(results[field]))
        elif field in metric.integer_array_fields:
            vals[field] = float(np.mean(results[field]))
        elif field in metric.float_fields:
            vals[field] = float(100 * float(results[field]))
        elif field in metric.integer_fields:
            vals[field] = int(results[field])

    # Include any extra keys returned by eval_sequence but not in the field lists
    for field, value in results.items():
        if field in vals:
            continue
        if isinstance(value, np.ndarray):
            vals[field] = float(np.mean(value))
        elif isinstance(value, (int, np.integer)):
            vals[field] = int(value)
        elif isinstance(value, (float, np.floating)):
            vals[field] = float(value)
        else:
            vals[field] = value

    return vals


def save_combined_metrics_csv(output_res, metrics_list, output_csv_path, gt_path=None, tracker_path=None):
    """Save evaluation results to a combined metrics CSV in wide format.

    dataset, tracker, and sequence columns are parsed from the input file paths
    when available, falling back to the evaluator's internal labels otherwise.
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

    for dataset_name, trackers in output_res.items():
        for tracker_name_eval, sequences in trackers.items():
            for seq_name_eval, metric_results in sequences.items():
                # Use parsed names when available, otherwise fall back to evaluator labels
                row = {
                    'dataset': tracker_dataset or gt_dataset or dataset_name,
                    'tracker': tracker_name or tracker_name_eval,
                    'sequence': tracker_seq or gt_seq or seq_name_eval,
                }
                for metric_name, metric_obj in metric_objects.items():
                    if metric_name not in metric_results:
                        continue
                    summary = get_metric_summary_values(metric_obj, metric_results[metric_name])
                    for field, value in summary.items():
                        row[f'{metric_name}_{field}'] = value
                rows.append(row)

    df = pd.DataFrame(rows)
    # Capitalise the first letter of each column title
    df.columns = [col[0].upper() + col[1:] if col else col for col in df.columns]
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
        'METRICS': ['HOTA', 'CLEAR', 'Identity', 'VACE'],
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
    save_combined_metrics_csv(output_res, metrics_list, output_csv, gt_path=gt_path, tracker_path=tracker_path)
