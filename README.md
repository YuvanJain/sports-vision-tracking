# Sports Vision Tracking

## Executive summary

This repository contains Day 1 detection benchmarks and final football and basketball detection/tracking workflows. The final football run uses **YOLOv8s** for people and the ball, **ByteTrack** for person IDs, a green-pitch filter, and temporal jersey-color votes for Team A (red), Team B (blue/purple), and referee kit (lime). Uncertain people retain `unknown_person`; ball detections have no persistent track ID. The final code is a **reconstructed V4-equivalent implementation**, not the exact recovered original V4 source.

## Architecture and tooling

Both pipelines use YOLOv8s because one small detector can handle people and ball candidates within a free Colab T4 budget. ByteTrack links person detections across frames without a separate re-identification model; this keeps inference practical but cannot prevent occlusion and camera-cut ID fragmentation. Football uses a green-surface pitch gate and temporal shirt-colour votes. Basketball uses camera-relative court evidence, upper-torso uniform evidence and temporal role/team decisions, with a separate referee path. Team labels are withheld when evidence is uncertain. The ball is detected per frame but has no persistent ID. Input frames are scaled for inference to fit memory; small or blurred balls are frequently missed. Neither pipeline has verified jersey OCR or shot outcomes.

## Repository map

- `notebooks/00_day1_benchmarking.ipynb`, `scripts/benchmark_models.py`, and `outputs/benchmarks/`: Day 1 benchmark and recorded results.
- `reference/v4_equivalent/`: reconstructed V4-equivalent 15-second and 60-second validation pipeline, config, notebook, and comparison utility.
- `reference/v4_final/`: continuous final-run script, config, Colab notebook, requirements, and detailed run instructions.
- `outputs/football/final_v4_equivalent_20261107_201332_839353/`: five chunks of tracking CSV/JSON, per-chunk metrics and pitch diagnostics, plus overall metrics.
- `docs/football_validation.md`: audit method, results, and limitations.
- `reference/basketball/submission_stage1e/`: frozen Stage 1E basketball tracking source, config, dependencies, export verifier, and GitHub Colab notebook.

Earlier root-level Day 2 script, config, and notebook remain in the project's history. For the published final result, use `reference/v4_final/` rather than the root-level trial code.

## Setup, installation and football inference

Select a fresh **T4 GPU** runtime and open `reference/v4_final/notebooks/02_final_football_10min.ipynb`. Package the contents of `reference/v4_final/` as `/content/sports_vision_v4_final.zip` with `scripts/`, `configs/`, `src/`, `requirements.txt`, and the notebook at the ZIP root. Upload that ZIP to Colab. The notebook mounts Google Drive and links `/content/drive/MyDrive/Sports Vision Tracking/Data/Football.mp4` into the extracted project; the video is not copied.

Run **code cell 1** for setup and dependency installation, then **code cell 2** for GPU, video, path, frame-count, and storage preflight. Check actual Google Drive quota separately because mounted filesystem free space may differ from cloud quota. **Code cell 3 starts the full 10-minute GPU job.** Code cell 4 verifies the completed files. Do not use *Run all* when you only intend to prepare the runtime. For a short validation, use `reference/v4_equivalent/notebooks/01_day2_football_tracking.ipynb` instead. A new football clip must be compatible with the fixed 25 FPS, 1080p, [1500,16500) workflow; set its Drive `VIDEO` path in the notebook before running.

The final script processes source frames **[1500, 16500)**, 01:00–11:00 at 25 FPS, with one detector, tracker, jersey-vote history, and pitch-gating state throughout. It writes five consecutive **3,000-frame** chunks. Each contains `annotated.mp4`, `tracks.csv`, `tracks.json`, `metrics.json`, `pitch_diagnostics.mp4`, and `pitch_diagnostics.jsonl`; the parent contains aggregate `metrics.json`. Video files stay on Drive. JSON has one entry per frame; CSV has one row per object or a frame-only row when no object is exported. Frame IDs and timestamps refer to the source video; boxes are source-pixel `[x_min, y_min, x_max, y_max]`.

If Colab interrupts a run, completed chunks may be inspected, but an exact resume is **not** supported: tracker and classifier state are not checkpointed. Restart the full interval in a new timestamped folder to retain globally consistent IDs. See `reference/v4_final/README.md` for the run procedure and limitations.

## Performance analysis and artifact links

The [final football Google Drive folder](https://drive.google.com/drive/folders/1X5BO_Nb4FRzkx1H-MjIbeAXoEVVFDssj) links the five annotated videos and final tracking artifacts. Selected football tracking exports are already tracked in this GitHub repository. Football MP4s, source videos, and model weights are not hosted here.

The final run exported **15,000 frames**, **1,950 globally unique person IDs**, and **2,950 ball-detected frames (19.67%)**. A read-only CSV audit found IDs continuing across all four chunk boundaries, **1,795 gaps** within tracks, **324 IDs** observed on at most five frames, **98 IDs** assigned both teams, **17 referee/team conflicts**, and **zero duplicate person frame/ID records**. Export continuity was verified; **ground-truth tracking and classification accuracy were not established**. Ball coverage measures model reports, not verified ball identity. The pitch filter can pass nearby staff or reject legitimate people, and jersey colors can confuse goalkeepers and referees. See [the validation report](docs/football_validation.md) for definitions and evidence.

## Basketball final workflow and output

Basketball uses YOLOv8s person/ball detections, ByteTrack person IDs, upper-torso jersey evidence (Team A white; Team B purple/blue with yellow), temporal team decisions, court participation evidence and separate referee handling. The [final basketball Drive folder](https://drive.google.com/drive/folders/1iH7mGQmoMuPolD_bvJMp6qPfcVsTzX9i) contains the completed annotated run. Its source interval is **[900, 18900)** at 30 FPS: 18,000 frames in five consecutive **3,600-frame (two-minute)** chunks. One tracker and role/team state persist across chunk boundaries. This folder's viewer access has not been independently checked from GitHub.

For Colab, open `reference/basketball/submission_stage1e/21_basketball_final_10min_github.ipynb` in a **fresh T4 runtime**. Cell 1 mounts Drive, clones this repository, links the existing `Data/Basketball.mp4` and `yolov8s.pt` into the checkout, and installs `requirements.txt`. Cell 2 checks CUDA, source video (30 FPS, 3840×2160, at least 18,900 frames), configuration and free space. Confirm the actual Drive quota separately: mounted free-space reports may differ. **Cell 3 alone starts inference** in a new timestamped folder under `My Drive/Sports Vision Tracking/outputs/basketball/`; Cell 4 checks the five saved chunks. Do not use *Run all* for setup. The source video is linked, not copied.

The included CLI supports a **compatible new video** at the same 30 FPS/resolution and with at least 18,900 source frames. Change the notebook's `VIDEO` path, or run from the extracted project root after linking existing weights:

```bash
python scripts/track_basketball_final.py --video Data/Basketball.mp4 --output-dir /content/drive/MyDrive/basketball_final_new_run --device cuda:0
```

The final interval is fixed in `configs/basketball_config.yaml` and the script; this is not an arbitrary-length inference command. Each chunk writes `annotated.mp4`, `tracks.csv`, `tracks.json`, `detections.jsonl` and `metrics.json`, with parent aggregate `metrics.json`. The final basketball `tracks.csv` and `metrics.json` files are available in both [GitHub](outputs/basketball/final_stage1e_20261116_115158_842463/) and the linked Google Drive folder. Full `tracks.json` and `detections.jsonl` exports and annotated basketball MP4s remain on Google Drive. Per-frame records carry source frame IDs/timestamps, person track IDs, class, source-pixel boxes and available team labels. `event` and `shot_outcome` are null because shot recognition was not validated. If Colab interrupts, preserve partial output and restart from frame 900 in a new folder; tracker/classifier state is not checkpointed, so IDs cannot resume identically.

## Failure modes

**Football.** The pitch and jersey heuristics can include sideline staff, miss legitimate boundary players, and confuse goalkeeper or referee kits. Export continuity across chunks was verified, but ground-truth ID accuracy was not measured. Ball detection covered 19.67% of frames; the detected ball's identity is unverified. Jersey-number OCR was not implemented. Track fragmentation remains substantial. 

**Basketball.** Role/team decisions can miss referees or confuse spectators and players, especially at cuts and close-ups. Tracking IDs and team labels can fragment or change. Ball detections are incomplete, and ball identity is not tracked persistently. **Made/Missed shot recognition is not implemented or validated.** These exports are model predictions, not ground-truth accuracy measurements. 

## Future work

Collect labeled, person-disjoint validation examples for both sports; add camera-aware geometry and stronger ball evidence before computing match statistics. Football needs verified jersey OCR if squad numbers are required. Basketball needs a validated hoop and ball-trajectory model before displaying Made/Missed outcomes. These are proposed upgrades, not included features.
