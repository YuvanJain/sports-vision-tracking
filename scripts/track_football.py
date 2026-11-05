"""Run a short football detection and tracking trial from the project root."""

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common.video_loader import VideoSegmentLoader, rescale_bbox

PERSON = 0
BALL = 32
FRAME_FIELDS = ("frame_id", "timestamp", "segment_frame_id", "segment_timestamp")
OBJECT_FIELDS = ("track_id", "class_name", "bbox", "team_id", "confidence", "detection_pass")
COLORS = {"A": (55, 55, 230), "B": (230, 100, 60), "referee": (70, 235, 100),
          "unknown": (180, 180, 180), "ball": (0, 220, 255)}


def clip_bbox(bbox, width, height):
    return np.clip(bbox, [0, 0, 0, 0], [width, height, width, height]).tolist()


class JerseyClassifier:
    """Combine torso color evidence over each ByteTrack ID."""

    def __init__(self, palette):
        self.ranges = {
            name: [(np.array(low, np.uint8), np.array(high, np.uint8)) for low, high in ranges]
            for name, ranges in palette.items()
        }
        self.history = {}

    def classify(self, frame, bbox, track_id, frame_id):
        x1, y1, x2, y2 = map(int, bbox)
        width, height = x2 - x1, y2 - y1
        torso = frame[y1 + int(.20 * height):y1 + int(.50 * height),
                      x1 + int(.25 * width):x1 + int(.75 * width)]
        scores = np.zeros(len(self.ranges))
        if min(torso.shape[:2]) >= 3:
            hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
            for index, ranges in enumerate(self.ranges.values()):
                mask = np.zeros(hsv.shape[:2], np.uint8)
                for low, high in ranges:
                    mask |= cv2.inRange(hsv, low, high)
                scores[index] = np.count_nonzero(mask) / mask.size

        previous, last_frame = self.history.get(track_id, (np.zeros_like(scores), frame_id - 1))
        smoothed = .65 ** (frame_id - last_frame) * previous + .35 * scores
        self.history[track_id] = (smoothed, frame_id)
        second, best = np.argsort(smoothed)[-2:]
        if smoothed[best] < .30 or smoothed[best] - smoothed[second] < .15:
            return "unknown"
        return tuple(self.ranges)[best]


class FootballDetector:
    def __init__(self, device, settings, ball_fallback):
        from ultralytics import YOLO
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.utils import IterableSimpleNamespace, YAML
        from ultralytics.utils.checks import check_yaml

        self.model = YOLO("yolov8s.pt")
        tracker_args = IterableSimpleNamespace(**YAML.load(check_yaml("bytetrack.yaml")))
        self.tracker = BYTETracker(tracker_args)
        self.predict_args = {"device": device, "half": device.startswith("cuda"),
                             "conf": float(tracker_args.track_low_thresh), "verbose": False}
        self.settings = settings
        self.ball_fallback = ball_fallback

    def detect(self, frame):
        boxes = self.model.predict(frame, classes=[PERSON, BALL], imgsz=self.settings["imgsz"],
                                   **self.predict_args)[0].boxes.cpu().numpy()
        # Age the tracker on every frame, including frames with no person detections.
        people = self.tracker.update(boxes[boxes.cls == PERSON])
        balls = boxes[(boxes.cls == BALL) & (boxes.conf >= self.settings["ball_confidence"])]
        detection_pass = "base"
        if not len(balls) and self.ball_fallback:
            balls = self.model.predict(frame, classes=[BALL], imgsz=self.settings["ball_fallback_imgsz"],
                                       **self.predict_args)[0].boxes.cpu().numpy()
            balls = balls[balls.conf >= self.settings["ball_confidence"]]
            detection_pass = "high_resolution"
        ball = None
        if len(balls):
            best = int(np.argmax(balls.conf))
            ball = (balls.xyxy[best], float(balls.conf[best]), detection_pass)
        return people, ball


def write_trial(loader, detector, classifier, output_dir, frame_count):
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    track_ids = set()
    ball_frames = Counter()
    width, height = loader.scaled_dims
    frames = 0
    with TemporaryDirectory(dir=output_dir) as work_dir:
        work = Path(work_dir)
        writer = cv2.VideoWriter(str(work / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                 loader.native_fps, (width, height))
        start = time.perf_counter()
        try:
            if not writer.isOpened():
                raise RuntimeError("OpenCV could not open the MP4 writer")
            with (work / "tracks.json").open("w", encoding="utf-8") as json_file, (
                work / "tracks.csv"
            ).open("w", newline="", encoding="utf-8") as csv_file:
                csv_writer = csv.DictWriter(csv_file, fieldnames=FRAME_FIELDS + OBJECT_FIELDS)
                csv_writer.writeheader()
                json_file.write("[\n")
                for packet in islice(loader.stream_frames(), frame_count):
                    frame = packet.frame
                    people, ball = detector.detect(frame)
                    records = []
                    overlays = []
                    for x1, y1, x2, y2, raw_id, score, _, _ in people:
                        track_id = int(raw_id)
                        bbox = clip_bbox([x1, y1, x2, y2], width, height)
                        label = classifier.classify(frame, bbox, track_id, packet.global_frame_idx)
                        class_name = "referee" if label == "referee" else "player"
                        team = None if class_name == "referee" else label
                        track_ids.add(track_id)
                        records.append((track_id, class_name, bbox, team, float(score), "base"))
                        caption = f"#{track_id} {label}" if label not in ("A", "B") else f"#{track_id} Team {label}"
                        overlays.append((bbox, caption, COLORS[label]))
                    if ball is not None:
                        raw_box, score, detection_pass = ball
                        bbox = clip_bbox(raw_box, width, height)
                        records.append((None, "ball", bbox, None, score, detection_pass))
                        overlays.append((bbox, "Ball", COLORS["ball"]))
                        ball_frames[detection_pass] += 1

                    objects = []
                    for track_id, class_name, bbox, team, score, detection_pass in records:
                        objects.append(dict(zip(OBJECT_FIELDS, (
                            track_id, class_name, rescale_bbox(bbox, *packet.scale_factors),
                            team, round(score, 4), detection_pass
                        ))))
                        counts[team or class_name] += 1
                    metadata = dict(zip(FRAME_FIELDS, (
                        packet.global_frame_idx, packet.global_timestamp_sec,
                        packet.segment_frame_idx, packet.segment_timestamp_sec
                    )))
                    if frames:
                        json_file.write(",\n")
                    json.dump({**metadata, "objects": objects}, json_file)
                    for obj in objects or [{}]:
                        row = {**metadata, **obj}
                        if obj:
                            row["bbox"] = json.dumps(obj["bbox"])
                        csv_writer.writerow(row)
                    for bbox, caption, color in overlays:
                        x1, y1, x2, y2 = map(int, bbox)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, caption, (x1, max(16, y1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)
                    writer.write(frame)
                    frames += 1
                json_file.write("\n]\n")
        finally:
            writer.release()

        if frames != frame_count:
            raise RuntimeError(f"Decoded {frames} of {frame_count} requested frames")
        elapsed = time.perf_counter() - start
        metrics = {
            "frames_processed": frames,
            "elapsed_sec": round(elapsed, 3),
            "pipeline_fps": round(frames / elapsed, 2),
            "unique_person_track_ids": len(track_ids),
            "object_rows_by_label": dict(counts),
            "ball_frames_by_pass": dict(ball_frames),
            "ball_frame_coverage_pct": round(100 * ball_frames.total() / frames, 2),
        }
        (work / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        for file in work.iterdir():
            file.replace(output_dir / file.name)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--seconds", type=float, default=15)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ball-fallback", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be positive and finite")
    config = yaml.safe_load((ROOT / "configs/football_config.yaml").read_text(encoding="utf-8"))
    settings = config["tracking"]
    video = args.video or ROOT / config["video"]["path"]
    start = config["segment"]["start_frame"] if args.start_frame is None else args.start_frame
    output_dir = args.output_dir or ROOT / settings["output_dir"]
    with VideoSegmentLoader(video, start_frame=start) as loader:
        frame_count = round(args.seconds * loader.native_fps)
        if not 0 < frame_count <= loader.expected_frames:
            parser.error("requested interval must contain at least one frame and fit within the video")
        detector = FootballDetector(args.device, settings, args.ball_fallback)
        classifier = JerseyClassifier(settings["jersey_hsv"])
        print(json.dumps(write_trial(loader, detector, classifier, output_dir, frame_count), indent=2))


if __name__ == "__main__":
    main()
