"""Compare existing V4 exports. Runs only when explicitly invoked in Colab."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def differences(expected, actual, path=""):
    if type(expected) is not type(actual) and not (
        isinstance(expected, (int, float)) and isinstance(actual, (int, float))
    ):
        return [{"path": path, "original": expected, "candidate": actual}]
    if isinstance(expected, dict):
        result = []
        for key in sorted(expected.keys() | actual.keys()):
            if key not in expected or key not in actual:
                result.append({"path": f"{path}/{key}", "original": expected.get(key),
                               "candidate": actual.get(key), "missing_key": True})
            else:
                result.extend(differences(expected[key], actual[key], f"{path}/{key}"))
        return result
    if isinstance(expected, list):
        result = []
        if len(expected) != len(actual):
            result.append({"path": f"{path}/length", "original": len(expected), "candidate": len(actual)})
        for index, (left, right) in enumerate(zip(expected, actual)):
            result.extend(differences(left, right, f"{path}/{index}"))
        return result
    return [] if expected == actual else [{"path": path, "original": expected, "candidate": actual}]


def records(folder, filename):
    path = folder / filename
    if path.suffix == ".json":
        return json.loads(path.read_text())
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    grouped = {}
    with path.open(newline="") as source:
        for row in csv.DictReader(source):
            for key in ("frame_id", "segment_frame_id", "track_id"):
                row[key] = int(row[key]) if row[key] else None
            for key in ("timestamp", "segment_timestamp", "confidence"):
                row[key] = float(row[key]) if row[key] else None
            row["bbox"] = json.loads(row["bbox"]) if row["bbox"] else None
            row["team_id"] = row["team_id"] or None
            grouped.setdefault(row["frame_id"], []).append(row)
    return [{"frame_id": key, "rows": value} for key, value in grouped.items()]


def compare(original, candidate):
    report = {}
    for filename in ("tracks.json", "tracks.csv", "pitch_diagnostics.jsonl"):
        left, right = records(original, filename), records(candidate, filename)
        left_ids = [row["frame_id"] for row in left]
        right_ids = [row["frame_id"] for row in right]
        a = {row["frame_id"]: row for row in left}
        b = {row["frame_id"]: row for row in right}
        changed = []
        for frame in sorted(a.keys() | b.keys()):
            delta = differences(a.get(frame), b.get(frame))
            if delta:
                changed.append({"frame_id": frame, "differences": delta})
        report[filename] = {
            "original_frames": len(left), "candidate_frames": len(right),
            "duplicate_original_frames": [k for k, v in Counter(left_ids).items() if v > 1],
            "duplicate_candidate_frames": [k for k, v in Counter(right_ids).items() if v > 1],
            "frame_order_equal": left_ids == right_ids,
            "different_frames": len(changed), "details": changed,
        }
    left_metrics = json.loads((original / "metrics.json").read_text())
    right_metrics = json.loads((candidate / "metrics.json").read_text())
    report["metrics_differences"] = differences(left_metrics, right_metrics)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.original, args.candidate)
    # Exclusive creation protects previous comparisons and original outputs.
    with (args.candidate / "comparison.json").open("x") as output:
        json.dump(report, output, indent=2)
    for name, result in report.items():
        if isinstance(result, dict):
            print(name, "different frames:", result["different_frames"])
