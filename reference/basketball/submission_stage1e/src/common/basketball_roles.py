"""Basketball V4 appearance, participant-role, and team decisions.

These are cautious image heuristics, not a trained role classifier. All decisions
are reversible and are exported with their underlying evidence for review.
"""

from collections import Counter, defaultdict, deque

import cv2
import numpy as np


class AppearanceEvidence:
    """Read upper-torso color and vertically coherent stripe structure."""

    def __init__(self, config):
        self.config = config

    def observe(self, frame, box):
        x1, y1, x2, y2 = map(int, box)
        width, height = x2 - x1, y2 - y1
        ay, by = self.config["torso_y"]
        ax, bx = self.config["torso_x"]
        crop = frame[y1 + int(ay * height):y1 + int(by * height),
                     x1 + int(ax * width):x1 + int(bx * width)]
        result = {"crop_valid": False, "white_fraction": 0.0,
                  "purple_fraction": 0.0, "dark_fraction": 0.0,
                  "raw_team_vote": None, "strong_team_evidence": None,
                  "stripe_evidence": False, "stripe_transitions": 0,
                  "stripe_coverage": 0.0, "stripe_dark_runs": 0,
                  "stripe_light_runs": 0}
        if crop.size == 0 or min(crop.shape[:2]) < self.config["min_torso_pixels"]:
            return result

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        color = {}
        for name, ranges in self.config["jersey_hsv"].items():
            mask = np.zeros(hsv.shape[:2], np.uint8)
            for low, high in ranges:
                mask |= cv2.inRange(hsv, np.array(low, np.uint8), np.array(high, np.uint8))
            color[name] = float(np.count_nonzero(mask) / mask.size)
        white, purple = color["A"], color["B"]
        dark = float(np.mean(hsv[:, :, 2] <= self.config["referee_dark_value_max"]))
        vote = None
        if (purple >= self.config["team_b_min_purple"] and
                purple >= white * self.config["team_b_min_purple_to_white_ratio"]):
            vote = "B"
        elif white >= self.config["team_a_min_white"] and purple <= self.config["team_a_max_purple"]:
            vote = "A"
        strong_team = None
        if purple >= self.config["strong_team_b_purple"]:
            strong_team = "B"
        elif (white >= self.config["strong_team_a_white"] and
              dark <= self.config["strong_team_a_max_dark"] and
              purple <= self.config["team_a_max_purple"]):
            strong_team = "A"

        # A slightly wider shirt sample can contain several genuine stripes;
        # the narrower center above remains the team-color sample.
        ry1, ry2 = self.config["referee_torso_y"]
        rx1, rx2 = self.config["referee_torso_x"]
        referee_crop = frame[y1 + int(ry1 * height):y1 + int(ry2 * height),
                             x1 + int(rx1 * width):x1 + int(rx2 * width)]
        if referee_crop.size == 0 or min(referee_crop.shape[:2]) < self.config["min_torso_pixels"]:
            result.update(crop_valid=True, white_fraction=round(white, 3),
                          purple_fraction=round(purple, 3), dark_fraction=round(dark, 3),
                          raw_team_vote=vote, strong_team_evidence=strong_team)
            return result
        # Vertical referee stripes must remain dark/light through much of the
        # torso height and alternate in several nontrivial-width column bands.
        stripe_crop = cv2.resize(referee_crop, (64, 64), interpolation=cv2.INTER_AREA)
        stripe_hsv = cv2.cvtColor(stripe_crop[8:56, 6:58], cv2.COLOR_BGR2HSV)
        dark_mask = stripe_hsv[:, :, 2] <= self.config["referee_dark_value_max"]
        light_mask = ((stripe_hsv[:, :, 2] >= self.config["referee_light_value_min"]) &
                      (stripe_hsv[:, :, 1] <= self.config["referee_light_saturation_max"]))
        coherence = self.config["referee_column_coherence"]
        columns = np.where(dark_mask.mean(axis=0) >= coherence, -1,
                           np.where(light_mask.mean(axis=0) >= coherence, 1, 0))
        runs = []
        for index, value in enumerate(columns):
            value = int(value)
            if runs and runs[-1][0] == value:
                runs[-1][2] = index + 1
            else:
                runs.append([value, index, index + 1])
        bands = [run for run in runs if run[0] and
                 run[2] - run[1] >= self.config["referee_min_band_width"]]
        transitions = sum(left[0] != right[0] for left, right in zip(bands, bands[1:]))
        coverage = float(np.count_nonzero(columns) / len(columns))
        dark_runs = sum(run[0] == -1 for run in bands)
        light_runs = sum(run[0] == 1 for run in bands)
        max_band = max((run[2] - run[1] for run in bands), default=0)
        stripe = bool(
            transitions >= self.config["referee_min_transitions"] and
            dark_runs >= self.config["referee_min_dark_bands"] and
            light_runs >= self.config["referee_min_light_bands"] and
            coverage >= self.config["referee_min_coverage"] and
            max_band <= self.config["referee_max_band_width"] and
            dark_mask.mean() >= self.config["referee_min_dark_fraction"] and
            light_mask.mean() >= self.config["referee_min_light_fraction"] and
            strong_team is None)
        result.update(crop_valid=True, white_fraction=round(white, 3),
                      purple_fraction=round(purple, 3), dark_fraction=round(dark, 3),
                      raw_team_vote=vote, strong_team_evidence=strong_team,
                      stripe_evidence=stripe, stripe_transitions=int(transitions),
                      stripe_coverage=round(coverage, 3),
                      stripe_dark_runs=int(dark_runs), stripe_light_runs=int(light_runs))
        return result


class RoleClassifier:
    """Use a finite evidence window; never lock an ID into a role forever."""

    def __init__(self, config):
        self.config = config
        self.history = defaultdict(lambda: deque(maxlen=config["window"]))
        self.stripes = defaultdict(lambda: deque(maxlen=config["referee_window"]))
        self.first_seen = {}
        self.last_active = {}

    def observe(self, track_id, frame_id, court_reason, interior, distance,
                relative_foot, appearance):
        self.first_seen.setdefault(track_id, frame_id)
        near = distance < self.config["nonparticipant_min_distance_fraction"]
        on_court = court_reason == "court"
        far_outside = (court_reason == "outside_court" and not near)
        history = self.history[track_id]
        history.append((frame_id, on_court, bool(interior), far_outside, near, relative_foot))
        court_count = sum(row[1] for row in history)
        interior_count = sum(row[2] for row in history)
        far_count = sum(row[3] for row in history)
        recent_court = sum(row[1] for row in list(history)[-self.config["recent_court_window"]:])
        age = frame_id - self.first_seen[track_id] + 1
        positions = [row[5] for row in history]
        motion = max((np.hypot(x - positions[0][0], y - positions[0][1])
                      for x, y in positions), default=0.0)
        stripes = self.stripes[track_id]
        if appearance["crop_valid"]:
            stripes.append((frame_id, bool(appearance["stripe_evidence"]),
                            appearance["strong_team_evidence"] is not None))
        recent = [row for row in stripes if frame_id - row[0] < self.config["referee_window"]]
        stripe_votes = sum(row[1] for row in recent)
        team_conflicts = sum(row[2] for row in recent)
        last_stripe = max((row[0] for row in recent if row[1]), default=-10**9)

        # Far outside with no interior history is stronger evidence of a bench
        # or spectator position than clothing, but never suppresses tracking.
        if (age >= self.config["nonparticipant_min_age_frames"] and
                far_count / len(history) >= self.config["nonparticipant_outside_fraction"] and
                interior_count == 0):
            role, reason = "nonparticipant", "repeated_far_outside_court"
        elif (stripe_votes >= self.config["referee_min_votes"] and
              len(recent) >= self.config["referee_min_valid_crops"] and
              stripe_votes / len(recent) >= self.config["referee_min_vote_fraction"] and
              team_conflicts <= self.config["referee_max_team_conflicts"] and
              frame_id - last_stripe <= self.config["referee_recent_stripe_frames"] and
              (recent_court >= 1 or near)):
            role, reason = "referee", "coherent_stripes_and_court_proximity"
        elif stripe_votes >= 2 and near and team_conflicts == 0:
            role, reason = "uncertain", "partial_referee_evidence"
        elif (age >= self.config["min_age_frames"] and
              court_count >= self.config["min_court_observations"] and
              interior_count >= self.config["min_interior_observations"] and
              recent_court >= self.config["min_recent_court_observations"]):
            role, reason = "active_player", "repeated_court_interior"
        elif (near and court_count >= self.config["boundary_court_observations"] and
              frame_id - self.last_active.get(track_id, -10**9) <=
              self.config["previous_active_grace_frames"]):
            role, reason = "active_player", "previous_player_near_boundary"
        else:
            role, reason = "uncertain", "insufficient_role_evidence"
        if role == "active_player":
            self.last_active[track_id] = frame_id
        return role, reason, {"court_votes": court_count,
                              "interior_votes": interior_count,
                              "recent_court_votes": recent_court,
                              "far_outside_votes": far_count,
                              "stripe_votes": stripe_votes,
                              "strong_team_conflicts": team_conflicts,
                              "camera_relative_movement": round(float(motion), 3)}


class TeamVotes:
    """Only confirmed active players accumulate stable team votes."""

    def __init__(self, config):
        self.config = config
        self.history = defaultdict(lambda: deque(maxlen=config["vote_window"]))
        self.state = {}

    def observe(self, track_id, frame_id, role, raw_vote):
        if role in ("referee", "nonparticipant"):
            self.history.pop(track_id, None)
            self.state.pop(track_id, None)
            return None, 0.0
        if role != "active_player":
            return None, 0.0
        self.history[track_id].append((frame_id, raw_vote))
        recent = [name for at, name in self.history[track_id]
                  if frame_id - at <= self.config["vote_max_age_frames"] and name]
        if not recent:
            return None, 0.0
        leader, count = Counter(recent).most_common(1)[0]
        ratio = count / len(recent)
        current = self.state.get(track_id)
        if count >= self.config["min_votes"] and ratio >= self.config["vote_fraction"]:
            if current is None or current == leader or count >= self.config["switch_min_votes"]:
                current = leader
                self.state[track_id] = leader
            else:
                current = None
        elif current is not None and ratio < self.config["vote_fraction"]:
            current = None
        return current, round(ratio, 3)
