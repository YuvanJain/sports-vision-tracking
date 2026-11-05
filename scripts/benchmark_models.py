"""Model Benchmarking Suite for Sports Vision Tracking.

Evaluates YOLO candidate models (YOLOv8n, YOLOv8s, YOLO11n, YOLO11s) under strictly
identical conditions on 30-second clips of both Football and Basketball.
Measures:
  - Model Inference FPS
  - Total Pipeline End-to-End FPS
  - Peak GPU VRAM consumption (MB)
  - Raw detection counts for person (class 0) and sports ball (class 32)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import yaml
from ultralytics import YOLO

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from src.common.video_loader import VideoSegmentLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark lightweight YOLO models on sports clips.")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device to run inference on (cuda:0 or cpu)")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference input size (default: 640)")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold (default: 0.45)")
    parser.add_argument("--warmup", type=int, default=30, help="Number of warmup frames (default: 30)")
    parser.add_argument("--output_dir", type=str, default="outputs/benchmarks", help="Directory to save results")
    return parser.parse_args()


def run_single_benchmark(
    model_name: str,
    sport_name: str,
    video_path: str,
    start_frame: int,
    end_frame: int,
    target_resolution: tuple,
    device: str,
    imgsz: int,
    conf: float,
    iou: float,
    warmup_frames: int
) -> Dict[str, Any]:
    """Runs an identical benchmark run for one model on one sports clip."""
    # FIX: Normalize device string into a canonical torch.device object.
    # PyTorch CUDA functions (reset_peak_memory_stats, synchronize,
    # max_memory_allocated) require torch.device, int, or None -- not a raw string.
    device_obj = torch.device(device)
    is_cuda = (device_obj.type == "cuda") and torch.cuda.is_available()

    if is_cuda:
        # Ensure the index is always explicit (cuda:0, not bare cuda)
        if device_obj.index is None:
            device_obj = torch.device(f"cuda:{torch.cuda.current_device()}")
        torch.cuda.set_device(device_obj)
        torch.cuda.empty_cache()
        # Passes torch.device object -- no longer a plain string
        torch.cuda.reset_peak_memory_stats(device=device_obj)

    print(f"\n[{sport_name.upper()}] Loading model: {model_name} on {device_obj} (imgsz={imgsz}, conf={conf})...")

    model = YOLO(model_name)
    use_half = is_cuda

    # Allocate dummy array BEFORE predict_kwargs is built
    dummy_input = np.zeros((target_resolution[1], target_resolution[0], 3), dtype=np.uint8)
    predict_kwargs = {
        "source": dummy_input,
        "device": str(device_obj),
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
        "verbose": False
    }
    if use_half:
        predict_kwargs["half"] = True

    print(f"[{sport_name.upper()}] Warming up {model_name} with {warmup_frames} frames...")
    for _ in range(warmup_frames):
        _ = model.predict(**predict_kwargs)

    if is_cuda:
        torch.cuda.synchronize(device=device_obj)

    loader = VideoSegmentLoader(
        video_path=video_path,
        start_frame=start_frame,
        end_frame=end_frame,
        target_resolution=target_resolution
    )

    total_frames = 0
    inference_durations = []
    total_pipeline_start = time.perf_counter()
    person_detections = 0
    ball_detections = 0
    zero_ball_frames = 0

    print(f"[{sport_name.upper()}] Streaming [{start_frame}, {end_frame}) ({loader.expected_frames} frames)...")

    for packet in loader.stream_frames():
        total_frames += 1

        if is_cuda:
            torch.cuda.synchronize(device=device_obj)
        t0 = time.perf_counter()

        predict_kwargs["source"] = packet.frame
        results = model.predict(
            classes=[0, 32],
            **predict_kwargs
        )

        if is_cuda:
            torch.cuda.synchronize(device=device_obj)
        t1 = time.perf_counter()
        inference_durations.append(t1 - t0)

        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            num_persons = int((cls_ids == 0).sum())
            num_balls = int((cls_ids == 32).sum())
            person_detections += num_persons
            ball_detections += num_balls
            if num_balls == 0:
                zero_ball_frames += 1
        else:
            zero_ball_frames += 1

    total_pipeline_end = time.perf_counter()
    loader.close()

    total_pipeline_time = total_pipeline_end - total_pipeline_start
    total_inference_time = sum(inference_durations)
    inference_fps = total_frames / total_inference_time if total_inference_time > 0 else 0.0
    pipeline_fps = total_frames / total_pipeline_time if total_pipeline_time > 0 else 0.0

    if is_cuda:
        peak_vram_mb = torch.cuda.max_memory_allocated(device=device_obj) / (1024 ** 2)
    else:
        peak_vram_mb = 0.0

    result = {
        "model": model_name,
        "sport": sport_name,
        "device": str(device_obj),
        "frames_processed": total_frames,
        "imgsz": imgsz,
        "conf_threshold": conf,
        "inference_time_sec": round(total_inference_time, 3),
        "total_pipeline_time_sec": round(total_pipeline_time, 3),
        "inference_fps": round(inference_fps, 2),
        "pipeline_fps": round(pipeline_fps, 2),
        "peak_vram_mb": round(peak_vram_mb, 2),
        "person_detections_total": person_detections,
        "person_detections_per_frame": round(person_detections / total_frames, 2) if total_frames > 0 else 0,
        "ball_detections_total": ball_detections,
        "ball_detections_per_frame": round(ball_detections / total_frames, 3) if total_frames > 0 else 0,
        "zero_ball_frames_pct": round((zero_ball_frames / total_frames) * 100, 2) if total_frames > 0 else 0
    }

    print(f"[{sport_name.upper()} - {model_name}] Inference: {inference_fps:.1f} FPS | Pipeline: {pipeline_fps:.1f} FPS | VRAM: {peak_vram_mb:.1f} MB | Persons: {person_detections} | Balls: {ball_detections}")
    return result


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open("configs/base_config.yaml", "r") as f:
        base_cfg = yaml.safe_load(f)
    with open("configs/football_config.yaml", "r") as f:
        fb_cfg = yaml.safe_load(f)
    with open("configs/basketball_config.yaml", "r") as f:
        bb_cfg = yaml.safe_load(f)

    models_to_test = base_cfg["benchmark"]["models"]
    results: List[Dict[str, Any]] = []

    print("=================================================================")
    print("      SPORTS VISION TRACKING: DAY 1 MODEL BENCHMARK SUITE       ")
    print(f"Device: {args.device} | ImgSz: {args.imgsz} | Conf: {args.conf} | IoU: {args.iou}")
    print("=================================================================")

    fb_sample = fb_cfg["benchmark_sample"]
    for model_name in models_to_test:
        res = run_single_benchmark(
            model_name=model_name,
            sport_name="football",
            video_path=fb_cfg["video"]["path"],
            start_frame=fb_sample["start_frame"],
            end_frame=fb_sample["end_frame"],
            target_resolution=tuple(fb_cfg["video"]["resolution"]),
            device=args.device,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            warmup_frames=args.warmup
        )
        results.append(res)

    bb_sample = bb_cfg["benchmark_sample"]
    for model_name in models_to_test:
        res = run_single_benchmark(
            model_name=model_name,
            sport_name="basketball",
            video_path=bb_cfg["video"]["path"],
            start_frame=bb_sample["start_frame"],
            end_frame=bb_sample["end_frame"],
            target_resolution=tuple(bb_cfg["video"]["inference_resolution"]),
            device=args.device,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            warmup_frames=args.warmup
        )
        results.append(res)

    df = pd.DataFrame(results)
    csv_path = Path(args.output_dir) / "benchmark_results.csv"
    json_path = Path(args.output_dir) / "benchmark_summary.json"
    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=================================================================")
    print("                   FINAL BENCHMARK SUMMARY TABLE                 ")
    print("=================================================================")
    cols_to_show = [
        "sport", "model", "inference_fps", "pipeline_fps", "peak_vram_mb",
        "person_detections_total", "ball_detections_total", "zero_ball_frames_pct"
    ]
    print(df[cols_to_show].to_markdown(index=False))
    print(f"\nFull benchmark saved to: {csv_path}")


if __name__ == "__main__":
    main()
