"""Run a short football detection and tracking trial from the project root."""

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter
from collections import deque
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


class PitchFilter:
    """Estimate the visible playing surface afresh for each camera frame."""

    def __init__(self, settings):
        self.settings = settings

    def mask(self, frame):
        config = self.settings["pitch"]
        height, width = frame.shape[:2]
        small = cv2.resize(frame, (width // 4, height // 4))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, np.array(config["hsv_low"], np.uint8),
                            np.array(config["hsv_high"], np.uint8))
        green = cv2.morphologyEx(green, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(green)
        if count <= 1:
            return None
        # Prefer the connected surface visible in the lower central image.
        # This rejects separate background fields when a walkway splits them.
        lower = labels[int(labels.shape[0] * .75):, int(labels.shape[1] * .2):int(labels.shape[1] * .8)]
        votes = np.bincount(lower.ravel(), minlength=count)
        votes[0] = 0
        component = int(np.argmax(votes))
        if not votes[component]:
            return None
        if stats[component, cv2.CC_STAT_AREA] < green.size * config["min_area_fraction"]:
            return None
        surface = np.uint8(labels == component) * 255
        surface = cv2.dilate(surface, np.ones((config["margin_cells"],) * 2, np.uint8))
        return cv2.resize(surface, (width, height), interpolation=cv2.INTER_NEAREST)

    def person_allowed(self, surface, bbox):
        if surface is None:
            return True
        height, width = surface.shape
        x1, y1, x2, y2 = bbox
        x = int(np.clip((x1 + x2) / 2, 0, width - 1))
        y = int(np.clip(y2, 0, height - 1))
        return bool(surface[y, x])


class JerseyClassifier:
    """Combine torso color evidence over each ByteTrack ID."""

    def __init__(self, palette, settings=None):
        self.ranges = {
            name: [(np.array(low, np.uint8), np.array(high, np.uint8)) for low, high in ranges]
            for name, ranges in palette.items()
        }
        self.history = {}
        self.settings = settings or {}

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

        # Require repeated independent observations; weak crops contribute no vote.
        second, best = np.argsort(scores)[-2:]
        vote = None
        if scores[best] >= self.settings.get("min_color_fraction", .30) and (
            scores[best] - scores[second] >= self.settings.get("min_color_margin", .15)
        ):
            vote = tuple(self.ranges)[best]
        history = self.history.setdefault(track_id, deque(maxlen=30))
        history.append((frame_id, vote))
        recent = [label for time_id, label in history
                  if frame_id - time_id <= 25 and label is not None]
        if len(recent) < self.settings.get("min_votes", 5):
            return "unknown"
        label, count = Counter(recent).most_common(1)[0]
        # Conflicting evidence stays visible as unknown instead of permanently
        # locking a potentially wrong identity or role.
        return label if count / len(recent) >= self.settings.get("vote_fraction", .85) else "unknown"


class FootballDetector:
    def __init__(self, device, settings, ball_fallback):
        from ultralytics import YOLO
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.utils import IterableSimpleNamespace, YAML
        from ultralytics.utils.checks import check_yaml

        self.model = YOLO("yolov8s.pt")
        tracker_args = IterableSimpleNamespace(**YAML.load(check_yaml("bytetrack.yaml")))
        for key, value in settings["bytetrack"].items():
            setattr(tracker_args, key, value)
        self.tracker = BYTETracker(tracker_args)
        self.predict_args = {"device": device, "half": device.startswith("cuda"),
                             "conf": float(tracker_args.track_low_thresh),
                             "iou": settings["nms_iou"], "verbose": False}
        self.settings = settings
        self.ball_fallback = ball_fallback
        self.pitch = PitchFilter(settings)
        self.frame_index = 0
        self.recent_pitch_boxes = deque()
        self.last_diagnostics = {}

    def detect(self, frame):
        boxes = self.model.predict(frame, classes=[PERSON, BALL], imgsz=self.settings["imgsz"],
                                   **self.predict_args)[0].boxes.cpu().numpy()
        surface = self.pitch.mask(frame)
        person_boxes = boxes[boxes.cls == PERSON]
        self.frame_index += 1
        grace = self.settings["pitch"]["boundary_grace_frames"]
        while self.recent_pitch_boxes and self.frame_index - self.recent_pitch_boxes[0][0] > grace:
            self.recent_pitch_boxes.popleft()
        decisions = []
        allowed = []
        for bbox in person_boxes.xyxy:
            if not self.settings["pitch"]["enabled"]:
                reason = "filter_disabled"
            elif surface is None:
                reason = "mask_uncertain"
            elif self.pitch.person_allowed(surface, bbox):
                reason = "pitch"
                self.recent_pitch_boxes.append((self.frame_index, bbox.copy()))
            elif any(self._overlap(bbox, prior) >= self.settings["pitch"]["boundary_grace_iou"]
                     for _, prior in self.recent_pitch_boxes):
                reason = "boundary_grace"
            else:
                reason = "outside_pitch"
            allowed.append(reason != "outside_pitch")
            decisions.append({"bbox": [float(v) for v in bbox], "reason": reason})
        person_boxes = person_boxes[np.asarray(allowed, dtype=bool)]
        self.last_diagnostics = {"mask": surface, "decisions": decisions,
                                 "accepted_reasons": [reason["reason"] for reason, keep in zip(decisions, allowed) if keep],
                                 "mask_status": "estimated" if surface is not None else "uncertain"}
        # Age the tracker on every frame, including frames with no person detections.
        people = self.tracker.update(person_boxes)
        balls = boxes[(boxes.cls == BALL) & (boxes.conf >= self.settings["ball_confidence"])]
        detection_pass = "base"
        if not len(balls) and self.ball_fallback:
            balls = self.model.predict(frame, classes=[BALL], imgsz=self.settings["ball_fallback_imgsz"],
                                       **self.predict_args)[0].boxes.cpu().numpy()
            balls = balls[balls.conf >= self.settings["ball_confidence"]]
            detection_pass = "high_resolution"
        # Sequential overlapping tiles preserve more pixels for distant balls.
        # Only run after both full-frame passes fail. Export model observations only.
        if not len(balls) and self.ball_fallback and self.settings.get("ball_tiles", False):
            height, width = frame.shape[:2]
            tile_w, tile_h = int(width * .6), int(height * .6)
            candidates = []
            for top in (0, height - tile_h):
                for left in (0, width - tile_w):
                    crop = frame[top:top + tile_h, left:left + tile_w]
                    found = self.model.predict(
                        crop, classes=[BALL], imgsz=self.settings["ball_fallback_imgsz"],
                        **self.predict_args
                    )[0].boxes.cpu().numpy()
                    for bbox, confidence in zip(found.xyxy, found.conf):
                        if confidence >= self.settings["ball_tile_confidence"]:
                            mapped = bbox + np.array([left, top, left, top])
                            candidates.append((mapped, float(confidence), "tile"))
            if candidates:
                return people, max(candidates, key=lambda item: item[1])
        ball = None
        if len(balls):
            best = int(np.argmax(balls.conf))
            ball = (balls.xyxy[best], float(balls.conf[best]), detection_pass)
        return people, ball

    @staticmethod
    def _overlap(a, b):
        x1, y1 = np.maximum(a[:2], b[:2])
        x2, y2 = np.minimum(a[2:], b[2:])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return intersection / max(1, area_a + area_b - intersection)


def write_trial(loader, detector, classifier, output_dir, frame_count):
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Choose a new empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    track_ids = set()
    ball_frames = Counter()
    pitch_evidence = Counter()
    width, height = loader.scaled_dims
    frames = 0
    with TemporaryDirectory(dir=output_dir) as work_dir:
        work = Path(work_dir)
        writer = cv2.VideoWriter(str(work / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                 loader.native_fps, (width, height))
        diagnostic_writer = cv2.VideoWriter(str(work / "pitch_diagnostics.mp4"),
                                            cv2.VideoWriter_fourcc(*"mp4v"),
                                            loader.native_fps, (width, height))
        start = time.perf_counter()
        try:
            if not writer.isOpened() or not diagnostic_writer.isOpened():
                raise RuntimeError("OpenCV could not open the MP4 writer")
            with (work / "tracks.json").open("w", encoding="utf-8") as json_file, (
                work / "tracks.csv"
            ).open("w", newline="", encoding="utf-8") as csv_file, (
                work / "pitch_diagnostics.jsonl"
            ).open("w", encoding="utf-8") as diagnostic_file:
                csv_writer = csv.DictWriter(csv_file, fieldnames=FRAME_FIELDS + OBJECT_FIELDS)
                csv_writer.writeheader()
                json_file.write("[\n")
                for packet in islice(loader.stream_frames(), frame_count):
                    frame = packet.frame
                    people, ball = detector.detect(frame)
                    diagnostics = getattr(detector, "last_diagnostics", {})
                    if diagnostics:
                        mask = diagnostics["mask"]
                        debug_frame = frame.copy()
                        if mask is not None:
                            tint = np.zeros_like(debug_frame)
                            tint[:, :, 1] = mask
                            debug_frame = cv2.addWeighted(debug_frame, 0.8, tint, 0.2, 0)
                        for item in diagnostics["decisions"]:
                            x1, y1, x2, y2 = map(int, item["bbox"])
                            reason = item["reason"]
                            color = (0, 0, 255) if reason == "outside_pitch" else (0, 255, 0)
                            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(debug_frame, reason, (x1, max(16, y1 - 4)),
                                        cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
                        diagnostic_writer.write(debug_frame)
                        diagnostic_file.write(json.dumps({
                            "frame_id": packet.global_frame_idx,
                            "mask_status": diagnostics["mask_status"],
                            "detections": diagnostics["decisions"]
                        }) + "\n")
                    records = []
                    overlays = []
                    for x1, y1, x2, y2, raw_id, score, _, detection_index in people:
                        track_id = int(raw_id)
                        bbox = clip_bbox([x1, y1, x2, y2], width, height)
                        label = classifier.classify(frame, bbox, track_id, packet.global_frame_idx)
                        accepted = diagnostics.get("accepted_reasons", [])
                        if int(detection_index) < len(accepted) and accepted[int(detection_index)] == "pitch":
                            pitch_evidence[track_id] += 1
                        eligible = pitch_evidence[track_id] >= detector.settings["pitch"]["min_confirmed_frames"]
                        class_name = ("referee" if eligible and label == "referee" else
                                      "player" if eligible and label in ("A", "B") else "unknown_person")
                        team = label if class_name == "player" else None
                        track_ids.add(track_id)
                        records.append((track_id, class_name, bbox, team, float(score), "base"))
                        caption = f"#{track_id} {label}" if eligible else f"#{track_id} role uncertain"
                        overlays.append((bbox, caption, COLORS[label] if eligible else COLORS["unknown"]))
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
            diagnostic_writer.release()

        if frames != frame_count:
            raise RuntimeError(f"Decoded {frames} of {frame_count} requested frames")
        elapsed = time.perf_counter() - start
        metrics = {
            "settings": getattr(detector, "settings", {}),
            "ball_fallback": getattr(detector, "ball_fallback", False),
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
        classifier = JerseyClassifier(settings["jersey_hsv"], settings["classification"])
        print(json.dumps(write_trial(loader, detector, classifier, output_dir, frame_count), indent=2))


if __name__ == "__main__":
    main()
