import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import yaml

from scripts.track_football import FootballDetector, JerseyClassifier, write_trial
from src.common.video_loader import FramePacket


class FootballTrackingTests(unittest.TestCase):
    def setUp(self):
        config = Path(__file__).resolve().parents[1] / "configs/football_config.yaml"
        self.settings = yaml.safe_load(config.read_text(encoding="utf-8"))["tracking"]
        self.classifier = JerseyClassifier(self.settings["jersey_hsv"])

    def test_jersey_labels_and_stale_evidence(self):
        box = [0, 0, 64, 64]
        for track_id, (hue, expected) in enumerate(
            [(178, "A"), (120, "B"), (49, "referee"), (35, "unknown")]
        ):
            hsv = np.full((64, 64, 3), (hue, 160, 150), dtype=np.uint8)
            frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            self.assertEqual(self.classifier.classify(frame, box, track_id, 1), expected)
        blank = np.zeros((64, 64, 3), dtype=np.uint8)
        self.assertEqual(self.classifier.classify(blank, box, 0, 20), "unknown")

    def test_ball_fallback_and_empty_person_frame(self):
        from ultralytics.engine.results import Boxes

        empty = Boxes(np.empty((0, 6)), (64, 64))
        fallback = Boxes(np.array([[1, 2, 4, 6, .8, 32], [3, 4, 6, 8, .4, 32]]), (64, 64))
        detector = FootballDetector.__new__(FootballDetector)
        detector.settings = self.settings
        detector.ball_fallback = True
        detector.predict_args = {}
        detector.tracker = Mock()
        detector.model = Mock()
        detector.model.predict.side_effect = [
            [SimpleNamespace(boxes=empty)], [SimpleNamespace(boxes=fallback)]
        ]
        _, ball = detector.detect(np.zeros((64, 64, 3), dtype=np.uint8))
        self.assertEqual(len(detector.tracker.update.call_args.args[0]), 0)
        self.assertEqual(detector.model.predict.call_args.kwargs["imgsz"], 1280)
        self.assertEqual(ball[0].tolist(), [1, 2, 4, 6])
        self.assertEqual(ball[2], "high_resolution")

    def test_exports_include_empty_frames_and_source_coordinates(self):
        packets = [
            FramePacket(np.zeros((64, 64, 3), np.uint8), 1500 + i, i, 60 + i / 25,
                        i / 25, (128, 128), (64, 64), (2, 2))
            for i in range(2)
        ]
        loader = SimpleNamespace(native_fps=25, scaled_dims=(64, 64),
                                 stream_frames=lambda: iter(packets))
        detector = Mock()
        detector.detect.side_effect = [
            ([[4, 4, 16, 28, 7, .9, 0, 0]], ([1, 2, 4, 6], .8, "high_resolution")),
            ([], None),
        ]
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            output = Path(directory)
            metrics = write_trial(loader, detector, self.classifier, output, 2)
            frames = json.loads((output / "tracks.json").read_text())
            self.assertEqual(frames[0]["objects"][0]["bbox"], [8, 8, 32, 56])
            self.assertEqual(frames[0]["objects"][0]["track_id"], 7)
            self.assertIsNone(frames[0]["objects"][1]["track_id"])
            self.assertEqual(frames[1]["objects"], [])
            with (output / "tracks.csv").open(newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual((rows[-1]["frame_id"], rows[-1]["class_name"]), ("1501", ""))
            self.assertEqual(metrics["ball_frame_coverage_pct"], 50)
            capture = cv2.VideoCapture(str(output / "annotated.mp4"))
            try:
                self.assertTrue(capture.read()[0])
                self.assertTrue(capture.read()[0])
                self.assertFalse(capture.read()[0])
            finally:
                capture.release()
