# Football tracking validation

The final run processed source frames **1500–16499 inclusive** (01:00–11:00 at 25 FPS) in one continuous YOLOv8s + ByteTrack session. Five consecutive chunks contain 3,000 frames each. The source is the **reconstructed V4-equivalent** pipeline, validated on 15-second and 60-second trials against the original V4 exports; it is not a recovered byte-for-byte copy of the original V4 implementation.

The local CSV audit found all 15,000 frame IDs, consistent global and segment timestamps, and zero duplicate person frame/track-ID records. IDs persisted on both sides of every chunk boundary: 18 at frames 4499/4500, 9 at 7499/7500, 13 at 10499/10500, and 18 at 13499/13500. This supports continuous ID numbering without a systematic boundary reset. It does not establish that every ID follows the correct person.

Before the final run, the reconstructed candidate was compared with preserved V4 results on the same intervals:

| Trial | Original V4 | Reconstructed V4-equivalent |
| --- | ---: | ---: |
| 15 s: person IDs / ball frames / unknown-person records / FPS | 49 / 215 / 928 / 5.74 | 49 / 216 / 928 / 6.19 |
| 60 s: person IDs / ball frames / unknown-person records / FPS | 176 / 692 / 5,151 / 7.35 | 175 / 692 / 5,121 / 8.10 |

These compact numerical comparisons support similar behavior on those intervals, not source-code identity or visual accuracy. The original V4 source was not recovered. The large frame-level `comparison.json` artifacts are not included in GitHub; the original validation outputs remain on Google Drive.

| Exported measure | Final result |
| --- | ---: |
| Globally unique person track IDs | 1,950 |
| Ball-detected frames | 2,950 / 15,000 (19.67%) |
| Gaps within person tracks | 1,795 |
| IDs observed on at most five frames | 324 |
| IDs assigned both Team A and Team B | 98 |
| IDs assigned both referee and team labels | 17 |
| Duplicate person frame/ID records | 0 |

These are statistics of **exported model predictions**, not ground-truth detection, team-classification, or tracking accuracy. A gap means an exported ID is absent between two observed frames; it is not necessarily a missed visible player. Multiple team labels on one ID indicate inconsistent classification or an identity switch, which cannot be separated without annotated ground truth. Ball coverage does not prove that the reported object is the real ball. Coach, staff, goalkeeper, referee, and touchline-player labels also need visual review. The five annotated videos and diagnostic videos remain in [Google Drive](https://drive.google.com/drive/folders/1X5BO_Nb4FRzkx1H-MjIbeAXoEVVFDssj).

Each final chunk publishes `tracks.csv`, `tracks.json`, `metrics.json`, and `pitch_diagnostics.jsonl`. The aggregate `metrics.json` is in the parent folder. The diagnostic JSONL records per-frame pitch-mask status and accepted or rejected detection reasons. The run retained one detector, ByteTrack instance, jersey-vote history, and pitch-gating state across all chunks. It did not serialize that state for exact continuation after an interruption.
