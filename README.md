# Sports Vision Tracking

This repository contains a Day 1 detection benchmark and a Day 2 football detection and tracking pipeline. The final football run uses **YOLOv8s** for people and the ball, **ByteTrack** for person IDs, a green-pitch filter, and temporal jersey-color votes for Team A (red), Team B (blue/purple), and referee kit (lime). Uncertain people retain `unknown_person`; ball detections have no persistent track ID. The final code is a **reconstructed V4-equivalent implementation**, not the exact recovered original V4 source.

## Repository map

- `notebooks/00_day1_benchmarking.ipynb`, `scripts/benchmark_models.py`, and `outputs/benchmarks/`: Day 1 benchmark and recorded results.
- `reference/v4_equivalent/`: reconstructed V4-equivalent 15-second and 60-second validation pipeline, config, notebook, and comparison utility.
- `reference/v4_final/`: continuous final-run script, config, Colab notebook, requirements, and detailed run instructions.
- `outputs/football/final_v4_equivalent_20261107_201332_839353/`: five chunks of tracking CSV/JSON, per-chunk metrics and pitch diagnostics, plus overall metrics.
- `docs/football_validation.md`: audit method, results, and limitations.

Earlier root-level Day 2 script, config, and notebook remain in the project's history. For the published final result, use `reference/v4_final/` rather than the root-level trial code.

## Run the football workflow in Google Colab

Select a fresh **T4 GPU** runtime and open `reference/v4_final/notebooks/02_final_football_10min.ipynb`. Package the contents of `reference/v4_final/` as `/content/sports_vision_v4_final.zip` with `scripts/`, `configs/`, `src/`, `requirements.txt`, and the notebook at the ZIP root. Upload that ZIP to Colab. The notebook mounts Google Drive and links `/content/drive/MyDrive/Sports Vision Tracking/Data/Football.mp4` into the extracted project; the video is not copied.

Run **code cell 1** for setup and dependency installation, then **code cell 2** for GPU, video, path, frame-count, and storage preflight. Check actual Google Drive quota separately because mounted filesystem free space may differ from cloud quota. **Code cell 3 starts the full 10-minute GPU job.** Code cell 4 verifies the completed files. Do not use *Run all* when you only intend to prepare the runtime. For a short validation, use `reference/v4_equivalent/notebooks/01_day2_football_tracking.ipynb` instead.

The final script processes source frames **[1500, 16500)**, 01:00–11:00 at 25 FPS, with one detector, tracker, jersey-vote history, and pitch-gating state throughout. It writes five consecutive **3,000-frame** chunks. Each contains `annotated.mp4`, `tracks.csv`, `tracks.json`, `metrics.json`, `pitch_diagnostics.mp4`, and `pitch_diagnostics.jsonl`; the parent contains aggregate `metrics.json`. Video files stay on Drive. JSON has one entry per frame; CSV has one row per object or a frame-only row when no object is exported. Frame IDs and timestamps refer to the source video; boxes are source-pixel `[x_min, y_min, x_max, y_max]`.

If Colab interrupts a run, completed chunks may be inspected, but an exact resume is **not** supported: tracker and classifier state are not checkpointed. Restart the full interval in a new timestamped folder to retain globally consistent IDs. See `reference/v4_final/README.md` for the run procedure and limitations.

## Results

The [final Google Drive output](https://drive.google.com/drive/folders/1X5BO_Nb4FRzkx1H-MjIbeAXoEVVFDssj) contains **five consecutive two-minute annotated football videos** and diagnostic videos. GitHub contains the practical structured tracking and diagnostic exports, not MP4s, source videos, or model weights.

The final run exported **15,000 frames**, **1,950 globally unique person IDs**, and **2,950 ball-detected frames (19.67%)**. A read-only CSV audit found IDs continuing across all four chunk boundaries, **1,795 gaps** within tracks, **324 IDs** observed on at most five frames, **98 IDs** assigned both teams, **17 referee/team conflicts**, and **zero duplicate person frame/ID records**. Export continuity was verified; **ground-truth tracking and classification accuracy were not established**. Ball coverage measures model reports, not verified ball identity. The pitch filter can pass nearby staff or reject legitimate people, and jersey colors can confuse goalkeepers and referees. See [the validation report](docs/football_validation.md) for definitions and evidence.
