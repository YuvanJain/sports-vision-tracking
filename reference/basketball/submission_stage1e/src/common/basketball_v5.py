"""V5 explainable basketball role evidence; no detector or tracker dependency.

The single-frame decision is also used by the offline reviewed-frame replay.
Runtime adds short, reversible per-track role votes. Neither path is a trained
participant classifier.
"""

from collections import Counter, defaultdict, deque

import cv2
import numpy as np

from .basketball_roles import AppearanceEvidence


class AppearanceV5(AppearanceEvidence):
    def observe(self, frame, box):
        result = super().observe(frame, box)
        x1, y1, x2, y2 = map(int, box)
        width, height = x2 - x1, y2 - y1
        ay, by = self.config["lower_uniform_y"]
        ax, bx = self.config["lower_uniform_x"]
        crop = frame[max(0, y1 + int(ay * height)):min(frame.shape[0], y1 + int(by * height)),
                     max(0, x1 + int(ax * width)):min(frame.shape[1], x1 + int(bx * width))]
        result.update(lower_valid=False, lower_white=0.0, lower_purple=0.0,
                      lower_yellow=0.0, lower_dark=0.0,
                      center_stripe_valid=False, center_stripe_dark=0.0,
                      center_stripe_light=0.0, center_stripe_coverage=0.0,
                      center_stripe_transitions=0)
        # Restrict the stripe check to the center of the detected torso. A
        # wider person box can contain another person's striped shirt at its
        # edge. The center check is evidence, not instance segmentation.
        ry1, ry2 = self.config["referee_torso_y"]
        cx1, cx2 = self.config["referee_center_x"]
        shirt = frame[max(0, y1 + int(ry1 * height)):min(frame.shape[0], y1 + int(ry2 * height)),
                      max(0, x1 + int(cx1 * width)):min(frame.shape[1], x1 + int(cx2 * width))]
        if shirt.size and min(shirt.shape[:2]) >= self.config["min_torso_pixels"]:
            hsv_shirt = cv2.cvtColor(cv2.resize(shirt, (64, 64), interpolation=cv2.INTER_AREA),
                                     cv2.COLOR_BGR2HSV)
            dark = hsv_shirt[:, :, 2] <= self.config["referee_dark_value_max"]
            light = ((hsv_shirt[:, :, 2] >= self.config["referee_light_value_min"]) &
                     (hsv_shirt[:, :, 1] <= self.config["referee_light_saturation_max"]))
            coherence = self.config["referee_center_column_coherence"]
            columns = np.where(dark.mean(axis=0) > coherence, -1,
                               np.where(light.mean(axis=0) > coherence, 1, 0))
            nonzero = columns[columns != 0]
            transitions = int(np.count_nonzero(nonzero[1:] != nonzero[:-1]))
            result.update(center_stripe_valid=True,
                          center_stripe_dark=round(float(dark.mean()), 3),
                          center_stripe_light=round(float(light.mean()), 3),
                          center_stripe_coverage=round(float(np.mean(columns != 0)), 3),
                          center_stripe_transitions=transitions)
        if crop.size == 0 or min(crop.shape[:2]) < self.config["min_torso_pixels"]:
            return result
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        result.update(lower_valid=True,
                      lower_white=round(float(np.mean((s <= 65) & (v >= 175))), 3),
                      lower_purple=round(float(np.mean((h >= 115) & (h <= 165) & (s >= 65))), 3),
                      lower_yellow=round(float(np.mean((h >= 15) & (h <= 40) & (s >= 90) & (v >= 75))), 3),
                      lower_dark=round(float(np.mean(v <= 85)), 3))
        return result


def classify_snapshot(appearance, court_reason, foot_y, playline, config):
    """Return role, team candidate, and reasons from one observation.

    Court floor color is supporting context. Shirt plus lower uniform evidence
    limits white-clothed staff being called players. A referee can be near the
    court boundary; the stripe cue requires multiple image properties.
    """
    a = appearance
    if not a["crop_valid"]:
        return "uncertain", None, "torso_crop_unavailable"
    if playline is None:
        return "uncertain", None, "court_geometry_unavailable"
    near_play = playline is not None and foot_y >= playline - config["boundary_margin"]
    near_court = court_reason in ("court", "boundary_grace")
    b_shirt = a["purple_fraction"] >= config["b_shirt_min"]
    a_shirt = a["white_fraction"] >= config["a_shirt_min"]
    b_bottom = max(a["lower_purple"], a["lower_yellow"]) >= config["b_lower_min"]
    a_bottom = a["lower_white"] >= config["a_lower_min"]
    referee_pattern = (a["center_stripe_valid"] and
                       a["center_stripe_transitions"] >= config["referee_center_min_transitions"] and
                       a["center_stripe_coverage"] >= config["referee_center_min_coverage"] and
                       a["center_stripe_dark"] >= config["referee_center_min_dark"] and
                       a["center_stripe_light"] >= config["referee_center_min_light"] and
                       a["lower_dark"] >= config["referee_min_lower_dark"] and
                       not b_shirt)
    if referee_pattern and near_court:
        return "referee", None, "center_stripes_and_court_support"
    if referee_pattern:
        return "uncertain", None, "stripes_without_court_support"
    if not near_play and not referee_pattern:
        if (b_shirt and b_bottom) or ((a_shirt or a_bottom) and a_bottom):
            return "uncertain", None, "uniform_near_sideline_or_bench"
        return "nonparticipant", None, "outside_dynamic_play_band"
    if b_shirt and b_bottom and near_court:
        return "active_player", "B", "purple_shirt_lower_uniform_court"
    if (a_shirt or a_bottom) and a_bottom and not b_shirt and near_court:
        return "active_player", "A", "white_lower_uniform_court"
    if not (b_shirt or a_shirt or a_bottom or referee_pattern):
        return "nonparticipant", None, "no_supported_uniform_or_stripes"
    return "uncertain", None, "insufficient_participant_evidence"


class RoleClassifierV5:
    """Short, reversible role history; uncertain at a conflicting boundary."""

    def __init__(self, config):
        self.config = config
        self.history = defaultdict(lambda: deque(maxlen=config["role_window"]))

    def observe(self, track_id, frame_id, appearance, court_reason, foot_y, playline):
        raw_role, candidate, instant_reason = classify_snapshot(
            appearance, court_reason, foot_y, playline, self.config)
        reason = instant_reason
        history = self.history[track_id]
        history.append((frame_id, raw_role, candidate))
        votes = Counter(role for _, role, _ in history if role != "uncertain")
        winner, count = votes.most_common(1)[0] if votes else ("uncertain", 0)
        conflicting = len(votes) > 1 and votes.most_common(2)[0][1] == votes.most_common(2)[1][1]
        if raw_role == "uncertain" or conflicting:
            role = "uncertain"
            reason = "conflicting_or_weak_temporal_evidence"
        elif count >= self.config["role_min_votes"] and winner == raw_role:
            role = winner
        else:
            role = "uncertain"
            reason = "role_vote_warmup"
        return role, candidate if role == "active_player" else None, reason, {
            "instant_role": raw_role, "instant_reason": instant_reason,
            "role_votes": dict(votes), "dynamic_playline_y": playline,
            "foot_y_fraction": round(foot_y, 3)}
