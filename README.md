# Sports Vision Tracking

Day 2 adds a football trial pipeline to the Day 1 detection benchmark. YOLOv8s detects people and the ball. ByteTrack associates people using motion and overlap and can recover short gaps through its low-confidence detection pass. A match-specific HSV palette classifies tracked torso crops as Team A (red), Team B (blue/purple), referee (lime), or unknown. Evidence accumulates per ID and decays over time.

## Run the football trials

In Colab, select a T4 GPU and open `notebooks/01_day2_football_tracking.ipynb`. Place the project and `Data/Football.mp4` under `/content/drive/MyDrive/Sports Vision Tracking`. The notebook installs dependencies and runs a 15-second smoke trial followed by a configurable 60-second trial. From the project root, the equivalent commands are

```sh
python -m pip install -r requirements.txt
python scripts/track_football.py --device cuda:0 --seconds 15 --output-dir outputs/football/smoke
python scripts/track_football.py --device cuda:0 --seconds 60 --ball-fallback --output-dir outputs/football/one_minute
```

The default start frame is 1500. Use `--video` and `--start-frame` for other clips and tune `tracking.jersey_hsv` in the football config for their kits. The optional ball fallback reruns YOLO at 1280 pixels on frames without a qualifying ball detection at 640 pixels. It increases compute and records the detection pass for comparison.

Each run writes `annotated.mp4`, `tracks.json`, `tracks.csv`, and `metrics.json`. JSON includes every frame with an objects array. CSV uses one row per object and a frame-only row when empty. Frame IDs and timestamps are source-relative, and boxes are source-pixel `[x_min, y_min, x_max, y_max]`. The ball is reported per frame with a null tracking ID. Coverage and throughput describe the run and require labeled footage to assess accuracy.

ByteTrack IDs can fragment after long occlusions or camera cuts. Team and referee labels can be uncertain or wrong when crops contain grass, spectators, or similarly colored kits. The final 01:00–11:00 segment is reserved for a later run.

Run the lightweight CPU checks with `python -B -m unittest discover -s tests`.
