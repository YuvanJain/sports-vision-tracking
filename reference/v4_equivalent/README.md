# Reconstructed V4-equivalent candidate

This isolated candidate reverses only the V5 changes documented in the conversation, starting from the preserved V5 script/config. It is not a recovered original ZIP and is not claimed byte-for-byte identical to V4. The 15-second and 60-second Colab runs were compared numerically with preserved V4 exports; their aggregate results were close, but source identity and visual accuracy were not established.

## Reconstruction decisions

- Removed the V5 eroded interior mask and participant state machine.
- Restored the V4 lower-central connected green component, morphology, dilation, and three-frame overlap grace. The undilated interior is not consulted.
- Restored cumulative pitch observations per track. Five pitch observations permit the existing temporal jersey vote to assign a player/referee label. Other tracks remain unknown_person.
- Removed participant_state from tracking exports and participant decisions from diagnostic JSONL.
- Preserved ByteTrack, YOLOv8s, color voting, ball passes, and diagnostic rendering from the V5 ancestor that retained them.
- Restored the recorded V4 tracking config, including imgsz 960, NMS IoU 0.60, buffer 60, ball thresholds 0.30 and 0.40, pitch area fraction 0.12, and margin 11.

The shared loader and requirements were copied from the current project; the conversation records no later changes to these after V4. The original runtime package versions, CUDA libraries and model checksum were not recorded. These can affect numerical repeatability. The V4 output metrics prove settings and observed results, not source identity.

## Colab validation

Open notebooks/01_day2_football_tracking.ipynb in a fresh T4 runtime. Upload sports_vision_v4_equivalent.zip to /content/, then execute setup and the function-definition cell. The Drive video must exist at /content/drive/MyDrive/Sports Vision Tracking/Data/Football.mp4.

Execute only the 15-second trial initially. It processes frames [1500,1875) with ball fallback enabled, writes a fresh timestamped smoke_v4_equivalent folder under Drive outputs/football, and compares against smoke_v4_20261106_150141_706053. The 60-second trial covers [1500,3000) and compares against one_minute_v4_20261106_150643_440214.

The comparator reads tracks.csv, tracks.json, pitch_diagnostics.jsonl and metrics.json from both folders. It writes comparison.json exclusively in the candidate output directory. Every differing frame includes field paths and original/candidate values. Numeric comparisons are exact, CSV numbers are parsed, and list order is preserved. Reordered detections can therefore cause differences even when geometry agrees; inspect those cases rather than interpreting every difference as a detection failure. Runtime/FPS differences are expected and are reported separately in metrics_differences.

Original smoke targets are 375 frames, 49 person IDs, 215 ball frames, 27 person-track gaps, and 935 rejected raw person detections. Original minute targets are 1,500 frames, 176 person IDs, 692 ball frames, 203 track gaps, and 1,929 rejected raw person detections. Summary matches are only a preliminary check. Require per-frame comparison and visual review of any disagreement before calling the candidate behaviorally equivalent on these intervals. A short-run match does not establish behavior on unseen footage.

The notebook also verifies the presence of the original output artifacts. If comparison fails, keep the newly generated trial and inspect the error; do not replace or edit original V4 files. There is no final ten-minute processing cell.
