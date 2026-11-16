"""Basketball V5.7 Stage 1E; bounded close-up player recognition. GPU untested."""

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common.video_loader import VideoSegmentLoader, rescale_bbox
from src.common.basketball_v55_team import TeamVotesV55
from src.common.basketball_v5 import AppearanceV5
from src.common.basketball_v52 import frame_context, snapshot_from_evidence
from src.common.basketball_stage1c import FIELDS_V57, FIELDS_1B, FIELDS_1C
from src.common.basketball_stage1e import PlayerStage1E as PlayerCloseupV57, TeamsStage1E as TeamsV57, FIELDS_1E
from src.common.basketball_scene import SceneEvidence

PERSON, BALL = 0, 32
FIELDS = ("frame_id", "timestamp", "segment_frame_id", "segment_timestamp",
          "track_id", "class_name", "bbox", "team_id", "confidence",
          "detection_pass", "court_evidence", "court_reason", "jersey_vote",
          "white_fraction", "purple_fraction", "team_vote_ratio",
          "participant_state", "participant_reason", "referee_evidence",
          "stripe_transitions", "stripe_coverage", "strong_team_evidence",
          "lower_white", "lower_purple", "lower_yellow", "lower_dark",
          "instant_role", "instant_reason", "team_candidate",
          "dynamic_playline_y", "foot_y_fraction",
          "center_stripe_dark", "center_stripe_light",
          "center_stripe_coverage", "center_stripe_transitions",
          "foreground_candidate", "foreground_votes", "cohort_count",
          "cohort_foot_p90", "camera_dx", "camera_dy", "motion_residual",
          "spatial_player", "player_votes", "referee_votes", "table_votes",
          "table_position", "table_stripe", "referee_appearance",
          "referee_duty", "role_change_reason", "foreground_without_cohort",
          "team_uniform_veto", "table_group_count", "ref_confirmed",
          "ref_last_strong_frame", "ref_weak_age_frames",
          "ref_player_contradiction_streak", "ref_recent_player_block",
          "ref_new_strong_votes", "ref_state_action",
          "cohort_reliable", "positive_player_spatial",
          "sustained_interior_player_evidence", "interior_player_streak",
          "established_player_persistence", "reliable_player_observations",
          "foreground_memory_active", "foreground_recovery_count",
          "ref_appearance_votes", "ref_duty_votes", "ref_duty_effective",
          "ref_duty_loss_streak", "ref_foreground_exception",
          "team_memory", "team_change_reason", "team_opposite_streak",
          "team_last_active_frame", "participation_grace_v56",
          "observation_gap_v56", "closeup_mask_unreliable",
          "closeup_geometry", "closeup_strong_stripe", "closeup_stripe_votes",
          "closeup_confirmed", "closeup_last_strong_frame",
          "closeup_team_veto_streak", "closeup_action")
COLORS = {"A": (240, 240, 240), "B": (190, 70, 160),
          "referee": (0, 220, 220), "unknown": (150, 150, 150),
          "table_official": (120, 220, 120),
          "ball": (0, 170, 255)}
FIELDS = (*FIELDS, *FIELDS_1E)


FIELDS = FIELDS + FIELDS_V57 + FIELDS_1B + FIELDS_1C

def clip_box(box, width, height):
    return np.clip(box, [0, 0, 0, 0], [width, height, width, height]).tolist()


class CourtEvidence:
    """Estimate the central playing floor for classification, not tracking."""

    def __init__(self, config):
        self.config = config
        self.history = defaultdict(lambda: deque(maxlen=config["evidence_window"]))
        self.last_supported = {}
        self.surface = None
        self.interior = None
        self.outside_distance = None
        self.centroid = (0.5, 0.5)
        self.status = "uncertain"
        self.playline = None

    def prepare(self, frame):
        height, width = frame.shape[:2]
        small = cv2.resize(frame, (width // 4, height // 4), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        wood_low, wood_high = self.config["wood_hsv"]
        wood = cv2.inRange(hsv, np.array(wood_low, np.uint8), np.array(wood_high, np.uint8))
        wood = cv2.morphologyEx(wood, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(wood)
        if count <= 1:
            self.surface, self.interior, self.outside_distance, self.status, self.playline = None, None, None, "uncertain", None
            return
        central = labels[int(labels.shape[0] * .46):int(labels.shape[0] * .73),
                         int(labels.shape[1] * .15):int(labels.shape[1] * .85)]
        votes = np.bincount(central.ravel(), minlength=count)
        votes[0] = 0
        component = int(np.argmax(votes))
        if not votes[component] or stats[component, cv2.CC_STAT_AREA] < wood.size * self.config["min_component_fraction"]:
            self.surface, self.interior, self.outside_distance, self.status, self.playline = None, None, None, "uncertain", None
            return
        selected = np.uint8(labels == component) * 255
        blue_low, blue_high = self.config["blue_hsv"]
        blue = cv2.inRange(hsv, np.array(blue_low, np.uint8), np.array(blue_high, np.uint8))
        nearby = cv2.dilate(selected, np.ones((15, 15), np.uint8))
        selected |= cv2.bitwise_and(blue, nearby)
        margin = max(1, self.config["floor_margin_pixels"] // 4)
        self.surface = cv2.dilate(selected, np.ones((margin * 2 + 1,) * 2, np.uint8))
        inner = max(1, self.config["interior_erode_pixels"] // 4)
        self.interior = cv2.erode(selected, np.ones((inner * 2 + 1,) * 2, np.uint8))
        self.outside_distance = cv2.distanceTransform(cv2.bitwise_not(self.surface), cv2.DIST_L2, 3)
        ys, xs = np.nonzero(selected)
        self.playline = float(np.quantile(ys / selected.shape[0], self.config["playline_quantile"]))
        self.centroid = (float(xs.mean() / selected.shape[1]), float(ys.mean() / selected.shape[0]))
        self.status = "estimated"

    def observe(self, frame, box, track_id, frame_id):
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = map(int, clip_box(box, width, height))
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            return False, 0.0, "invalid_box", False, 1.0, (0.0, 0.0)
        left = max(0, x1 + bw // 4)
        right = min(width, x2 - bw // 4)
        top = max(0, y2 - max(2, int(bh * self.config["foot_patch_height_fraction"])))
        bottom = min(height, y2 + max(3, int(bh * self.config["foot_patch_height_fraction"])))
        fraction = 0.0
        observed = False
        interior = False
        foot_x, foot_y = min((x1+x2)//8, max(0,width//4-1)), min(y2//4, max(0,height//4-1))
        distance = 1.0
        if self.surface is not None and right > left and bottom > top:
            patch = self.surface[top // 4:max(top // 4 + 1, (bottom + 3) // 4),
                                 left // 4:max(left // 4 + 1, (right + 3) // 4)]
            if patch.size:
                fraction = float(np.count_nonzero(patch) / patch.size)
                observed = fraction >= self.config["min_floor_fraction"]
                interior = bool(self.interior[foot_y, foot_x])
                distance = float(self.outside_distance[foot_y, foot_x] * 4 / height)
        history = self.history[track_id]
        history.append((frame_id, observed))
        if observed:
            self.last_supported[track_id] = frame_id
        count = sum(value for _, value in history)
        confirmed = count >= self.config["min_confirmed_frames"]
        grace = frame_id - self.last_supported.get(track_id, -10**9) <= self.config["boundary_grace_frames"]
        eligible = confirmed and (observed or grace)
        reason = "court" if observed else "boundary_grace" if eligible else "mask_uncertain" if self.surface is None else "outside_court"
        relative_foot = ((x1+x2)/2/width-self.centroid[0], y2/height-self.centroid[1])
        return eligible, fraction, reason, interior, distance, relative_foot




FINAL_FIELDS=FIELDS+("event","shot_outcome")
class ChunkSink:
    def __init__(self,root,index,fps,size):
        self.index=index;self.start=900+index*3600;self.path=root/f"chunk_{index+1:02d}";self.path.mkdir(exist_ok=False)
        self.writer=cv2.VideoWriter(str(self.path/"annotated.mp4"),cv2.VideoWriter_fourcc(*"mp4v"),fps,size)
        if not self.writer.isOpened():raise RuntimeError(f"Cannot write {self.path}/annotated.mp4")
        self.csv_handle=(self.path/"tracks.csv").open("w",newline="",encoding="utf-8")
        self.csv_writer=csv.DictWriter(self.csv_handle,fieldnames=FINAL_FIELDS);self.csv_writer.writeheader()
        self.json_handle=(self.path/"tracks.json").open("w",encoding="utf-8");self.json_handle.write("[\n")
        self.diagnostic_handle=(self.path/"detections.jsonl").open("w",encoding="utf-8")
        self.frames_processed=0;self.ids=set();self.ball_frames=0;self.visible=0;self.began=time.perf_counter()
    def count_frame(self,objects,visible):
        self.frames_processed+=1;self.visible+=visible
        for item in objects:
            if item["class_name"]=="ball":self.ball_frames+=1
            elif item["track_id"] is not None:self.ids.add(item["track_id"])
    def close(self,config):
        self.json_handle.write("\n]\n");self.json_handle.close();self.csv_handle.close();self.diagnostic_handle.close();self.writer.release()
        if self.frames_processed!=3600:raise RuntimeError(f"Incomplete chunk {self.index+1}: {self.frames_processed}/3600")
        result={"start_frame":self.start,"end_frame_exclusive":self.start+3600,"frames_processed":self.frames_processed,
          "unique_person_track_ids":len(self.ids),"ball_detected_frames":self.ball_frames,"visible_person_annotation_rows":self.visible,
          "elapsed_sec":round(time.perf_counter()-self.began,3),"shot_detection_implemented":False}
        (self.path/"metrics.json").write_text(json.dumps(result,indent=2))
        return result

def run(video, output, config, device, start_frame=None, end_frame=None):
    from ultralytics import YOLO
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.utils import IterableSimpleNamespace, YAML
    from ultralytics.utils.checks import check_yaml

    if output.exists():
        raise FileExistsError(f"Use a new timestamped output folder: {output}")
    output.mkdir(parents=True,exist_ok=False)
    settings = config["model"]
    tracker_args = IterableSimpleNamespace(**YAML.load(check_yaml("bytetrack.yaml")))
    for key, value in config["tracking"].items():
        setattr(tracker_args, key, value)
    tracker = BYTETracker(tracker_args)
    model = YOLO(str(ROOT / settings["weights"]))
    court = CourtEvidence(config["court_evidence"])
    appearance = AppearanceV5(config["classification"])
    roles = PlayerCloseupV57(config)
    teams = TeamsV57(config["classification"], config["team_memory_v55"])
    scene = SceneEvidence(config["scene_stage1b"])
    previous_centers = {}
    start_frame,end_frame=900,18900
    if config["final"]!={"start_frame":900,"end_frame_exclusive":18900,"chunk_frames":3600,"chunk_count":5}:
        raise ValueError("Final configuration must be [900,18900) in five chunks")
    counts = Counter()
    ids = set()
    active_ids = set()
    participant_counts = Counter()
    visible_person_rows = 0
    ball_frames = 0
    with VideoSegmentLoader(video,start_frame=900,end_frame=18900,target_resolution=tuple(config["video"]["inference_resolution"])) as loader:
        if abs(loader.native_fps-30)>.01 or loader.expected_frames!=18000:raise ValueError("Requires 30 FPS and 18000 source frames")
        width,height=loader.scaled_dims;frame_total=0;chunk_metrics=[];sink=None;began=time.perf_counter()
        try:
            for packet in loader.stream_frames():
                idx=(packet.global_frame_idx-900)//3600
                if sink is None or idx != sink.index:
                    if sink is not None: chunk_metrics.append(sink.close(config))
                    sink=ChunkSink(output,idx,loader.native_fps,(width,height))
                writer=sink.writer;csv_writer=sink.csv_writer;json_handle=sink.json_handle;diagnostic_handle=sink.diagnostic_handle
                before_visible=visible_person_rows
                frame = packet.frame
                scene_detail = scene.observe(frame, packet.global_frame_idx)
                if scene_detail["scene_cut"] or scene_detail["scene_quarantine"]:
                    court = CourtEvidence(config["court_evidence"])
                    previous_centers = {}
                court.prepare(frame)
                found = model.predict(frame, classes=[PERSON, BALL], imgsz=settings["imgsz"],
                                      conf=settings["confidence"], iou=settings["nms_iou"],
                                      device=device, half=device.startswith("cuda"), verbose=False)[0].boxes.cpu().numpy()
                raw_people = found[found.cls == PERSON]
                people = tracker.update(raw_people)
                balls = found[(found.cls == BALL) & (found.conf >= settings["ball_confidence"])]
                pass_name = "base"
                if not len(balls) and settings["ball_fallback"]:
                    more = model.predict(frame, classes=[BALL], imgsz=settings["ball_fallback_imgsz"],
                                         conf=settings["confidence"], iou=settings["nms_iou"],
                                         device=device, half=device.startswith("cuda"), verbose=False)[0].boxes.cpu().numpy()
                    balls = more[more.conf >= settings["ball_confidence"]]
                    pass_name = "high_resolution"
                objects = []
                overlays = []
                person_decisions = []
                prepared = []
                for x1, y1, x2, y2, raw_id, confidence, _, _ in people:
                    tid = int(raw_id)
                    box = clip_box([x1, y1, x2, y2], width, height)
                    _, floor_fraction, court_reason, interior, distance, relative_foot = court.observe(
                        frame, box, tid, packet.global_frame_idx)
                    evidence = appearance.observe(frame, box)
                    foot_y = box[3] / height
                    instant, raw_team, instant_reason = snapshot_from_evidence(
                        evidence, court_reason, foot_y, court.playline, config["participant"])
                    prepared.append({"track_id": tid, "frame_id": packet.global_frame_idx,
                                     "bbox": box, "width": width, "height": height,
                                     "confidence": confidence, "court_fraction": floor_fraction,
                                     "court_reason": court_reason, "court_mask_status": court.status,
                                     "interior": interior,
                                     "distance": distance, "relative_foot": relative_foot,
                                     "appearance": evidence, "foot": foot_y,
                                     "playline": court.playline, "instant": instant,
                                     "instant_reason": instant_reason, "raw_team": raw_team})
                context, previous_centers = frame_context(prepared, previous_centers)
                scene_detail.update(scene.floor_evidence(frame, prepared))
                context.update(scene_detail)
                for observation in prepared:
                    observation["scene_person"] = scene.person(observation)
                roles.begin_frame(prepared, context)
                for item in prepared:
                    tid, box = item["track_id"], item["bbox"]
                    evidence, foot_y = item["appearance"], item["foot"]
                    court_reason, floor_fraction = item["court_reason"], item["court_fraction"]
                    interior, distance = item["interior"], item["distance"]
                    state, team_candidate, participant_reason, role_detail = roles.observe(item, context)
                    # Unconfirmed participation cannot bank team votes across
                    # unrelated uncertain observations. Confirmed-player grace
                    # retains a short history without inventing a new jersey vote.
                    team_role = state
                    team, vote_ratio = teams.observe_with_detail(
                        tid, packet.global_frame_idx, team_role, team_candidate, role_detail)
                    role = ("referee" if state == "referee" else
                            "player" if state == "active_player" else
                            "table_official" if state == "table_official" else
                            "nonparticipant" if state == "nonparticipant" else "unknown_person")
                    ids.add(tid)
                    participant_counts[state] += 1
                    if state == "active_player":
                        active_ids.add(tid)
                    counts[team or role] += 1
                    objects.append({"track_id": tid, "class_name": role,
                                    "bbox": rescale_bbox(box, *packet.scale_factors),
                                    "team_id": team, "confidence": round(float(item["confidence"]), 4),
                                    "detection_pass": "base", "court_evidence": round(floor_fraction, 3),
                                    "court_reason": court_reason,
                                    "jersey_vote": evidence["raw_team_vote"],
                                    "white_fraction": evidence["white_fraction"],
                                    "purple_fraction": evidence["purple_fraction"],
                                    "team_vote_ratio": vote_ratio,
                                    "participant_state": state,
                                    "participant_reason": participant_reason,
                                    "referee_evidence": evidence["stripe_evidence"],
                                    "stripe_transitions": evidence["stripe_transitions"],
                                    "stripe_coverage": evidence["stripe_coverage"],
                                    "strong_team_evidence": evidence["strong_team_evidence"]})
                    objects[-1].update({"lower_white": evidence["lower_white"],
                                        "lower_purple": evidence["lower_purple"],
                                        "lower_yellow": evidence["lower_yellow"],
                                        "lower_dark": evidence["lower_dark"],
                                        "instant_role": item["instant"],
                                        "instant_reason": item["instant_reason"],
                                        "team_candidate": team_candidate,
                                        "dynamic_playline_y": court.playline,
                                        "foot_y_fraction": round(foot_y, 3),
                                        "center_stripe_dark": evidence["center_stripe_dark"],
                                        "center_stripe_light": evidence["center_stripe_light"],
                                        "center_stripe_coverage": evidence["center_stripe_coverage"],
                                        "center_stripe_transitions": evidence["center_stripe_transitions"],
                                        **role_detail, **teams.last_decision[tid]})
                    person_decisions.append({"track_id": tid, "court_reason": court_reason,
                                             "court_fraction": round(floor_fraction, 3),
                                             "court_interior": interior,
                                             "court_distance_fraction": round(distance, 3),
                                             "instant_role": item["instant"],
                                             "instant_reason": item["instant_reason"],
                                             **role_detail, **teams.last_decision[tid], **evidence,
                                             "team_candidate": team_candidate,
                                             "team_vote_ratio": vote_ratio,
                                             "participant_state": state,
                                             "participant_reason": participant_reason,
                                             "exported_class": role, "exported_team": team,
                                             **item["scene_person"]})
                    show = state in ("active_player", "referee") or (
                        state == "uncertain" and config["participant"]["show_uncertain_near_court"] and
                        distance < config["participant"]["nonparticipant_min_distance_fraction"])
                    if show:
                        caption = f"#{tid} {team or role}"
                        overlays.append((box, caption, COLORS.get(team or role, COLORS["unknown"])))
                        visible_person_rows += 1
                ball_candidates = []
                for box, confidence in zip(balls.xyxy, balls.conf):
                    if float(confidence) >= settings["ball_confidence"]:
                        ball_candidates.append({"bbox": rescale_bbox(clip_box(box, width, height), *packet.scale_factors),
                                                "confidence": round(float(confidence), 4), "detection_pass": pass_name})
                if ball_candidates:
                    best = max(ball_candidates, key=lambda item: item["confidence"])
                    objects.append({"track_id": None, "class_name": "ball", "bbox": best["bbox"],
                                    "team_id": None, "confidence": best["confidence"],
                                    "detection_pass": pass_name, "court_evidence": None,
                                    "court_reason": None, "jersey_vote": None,
                                    "white_fraction": None, "purple_fraction": None,
                                    "team_vote_ratio": None, "participant_state": None,
                                    "participant_reason": None, "referee_evidence": None,
                                    "stripe_transitions": None, "stripe_coverage": None,
                                    "strong_team_evidence": None, "lower_white": None,
                                    "lower_purple": None, "lower_yellow": None,
                                    "lower_dark": None, "instant_role": None,
                                    "instant_reason": None, "team_candidate": None,
                                    "dynamic_playline_y": None,
                                    "foot_y_fraction": None,
                                    "center_stripe_dark": None,
                                    "center_stripe_light": None,
                                    "center_stripe_coverage": None,
                                    "center_stripe_transitions": None})
                    # Convert source-pixel coordinates to annotated-frame pixels.
                    sx, sy = packet.scale_factors
                    bx = [best["bbox"][0] / sx, best["bbox"][1] / sy,
                          best["bbox"][2] / sx, best["bbox"][3] / sy]
                    overlays.append((bx, "ball candidate", COLORS["ball"]))
                    ball_frames += 1
                    counts["ball"] += 1
                meta = {"frame_id": packet.global_frame_idx,
                        "timestamp": packet.global_timestamp_sec,
                        "segment_frame_id": packet.segment_frame_idx,
                        "segment_timestamp": packet.segment_timestamp_sec}
                if sink.frames_processed:
                    json_handle.write(",\n")
                json.dump({**meta, "objects": objects,"event":None,"shot_outcome":None}, json_handle)
                for item in objects or [{}]:
                    row = {**meta, **item}
                    if item:
                        row["bbox"] = json.dumps(row["bbox"])
                    csv_writer.writerow({**row,"event":None,"shot_outcome":None})
                raw_person_candidates = [
                    {"bbox": rescale_bbox(clip_box(box, width, height), *packet.scale_factors),
                     "confidence": round(float(confidence), 4)}
                    for box, confidence in zip(raw_people.xyxy, raw_people.conf)]
                raw_ball_candidates = [
                    {"bbox": rescale_bbox(clip_box(box, width, height), *packet.scale_factors),
                     "confidence": round(float(confidence), 4)}
                    for box, confidence in zip(found[found.cls == BALL].xyxy,
                                               found[found.cls == BALL].conf)]
                diagnostic_handle.write(json.dumps({**meta, "raw_person_detections": len(raw_people),
                                                    "raw_person_candidates": raw_person_candidates,
                                                    "raw_ball_candidates": raw_ball_candidates,
                                                    "tracked_persons": len(people),
                                                    "court_mask_status": court.status,
                                                    "scene_evidence": scene_detail,
                                                    "person_decisions": person_decisions,
                                                    "ball_candidates": ball_candidates,"event":None,"shot_outcome":None}) + "\n")
                for box, text, color in overlays:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, text, (x1, max(16, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
                writer.write(frame)
                sink.count_frame(objects,visible_person_rows-before_visible)
                frame_total += 1
        finally:
            if sink is not None:
                if sink.frames_processed==3600:chunk_metrics.append(sink.close(config))
                else:
                    sink.json_handle.close();sink.csv_handle.close();sink.diagnostic_handle.close();sink.writer.release()
        if frame_total!=18000 or len(chunk_metrics)!=5:raise RuntimeError(f"Incomplete final run: {frame_total} frames, {len(chunk_metrics)} chunks")
        metrics={"version":"basketball_v5_7_stage1e_final","start_frame":900,"end_frame_exclusive":18900,
          "frames_processed":frame_total,"chunk_frames":3600,"chunk_count":5,"unique_person_track_ids":len(ids),
          "ball_detected_frames":ball_frames,"chunk_metrics":chunk_metrics,"elapsed_sec":round(time.perf_counter()-began,3),
          "shot_detection_implemented":False,"shot_outcomes":"not available; event and shot_outcome fields are null",
          "limitations":"Predicted tracking and roles require visual review; no made/missed shot recognition."}
        (output/"metrics.json").write_text(json.dumps(metrics,indent=2));(output/"source_manifest.json").write_text((ROOT/"source_manifest.json").read_text())
        return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--end-frame", type=int)
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "configs/basketball_config.yaml").read_text(encoding="utf-8"))
    print(json.dumps(run(args.video, args.output_dir, config, args.device,
                         args.start_frame, args.end_frame), indent=2))


if __name__ == "__main__":
    main()
