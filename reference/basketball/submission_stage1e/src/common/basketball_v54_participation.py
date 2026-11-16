"""V5.4 participation evidence after detection and tracking.

All positions are camera-relative. These rules are reversible heuristics, not a
trained participant model. They do not remove detections or alter ByteTrack.
"""

from collections import Counter, defaultdict, deque
from statistics import median

from .basketball_v5 import classify_snapshot


def frame_context(observations, previous_centers):
    """Describe the current playing cohort and common image translation."""
    feet = sorted(float(o["foot"]) for o in observations
                  if o["instant"] == "active_player" and o["court_reason"] == "court"
                  and float(o["foot"]) < 0.87)
    dx, dy = [], []
    centers = {}
    table_candidates = []
    for o in observations:
        box = o["bbox"]
        center = ((box[0] + box[2]) / (2 * o["width"]),
                  (box[1] + box[3]) / (2 * o["height"]))
        old = previous_centers.get(int(o["track_id"]))
        if old is not None:
            dx.append(center[0] - old[0])
            dy.append(center[1] - old[1])
        centers[int(o["track_id"])] = center
        a = o["appearance"]
        if (o["playline"] is not None and o["court_reason"] == "outside_court" and
                o["foot"] < o["playline"] - .10 and
                (box[3] - box[1]) / o["height"] < .11 and
                a["center_stripe_coverage"] >= .10 and
                a["center_stripe_dark"] >= .08 and
                a["center_stripe_light"] >= .08):
            table_candidates.append((center[0], o["foot"]))
    table_group = []
    for x, y in table_candidates:
        neighbors = [px for px, py in table_candidates if abs(px-x) <= .25 and abs(py-y) <= .08]
        if len(neighbors) > len(table_group):
            table_group = neighbors
    context = {"cohort_count": len(feet),
               "cohort_foot_median": median(feet) if feet else None,
               "cohort_foot_p90": feet[min(len(feet) - 1, int(len(feet) * .9))] if feet else None,
               "camera_dx": median(dx) if dx else 0.0,
               "camera_dy": median(dy) if dy else 0.0,
               "table_group_count": len(table_group),
               "table_x_min": min(table_group) if table_group else None,
               "table_x_max": max(table_group) if table_group else None}
    return context, centers


class RoleClassifierV54Participation:
    def __init__(self, config):
        self.config = config
        self.history = defaultdict(lambda: deque(maxlen=config["evidence_window"]))
        self.previous_center = {}
        self.last_active = {}
        self.last_team = {}
        self.last_persistent_foreground = {}
        self.foreground_recovery = defaultdict(int)
        self.reliable_player_observations = defaultdict(int)
        self.last_reliable_player_observation = {}
        self.interior_player_streak = defaultdict(int)
        self.last_seen = {}

    def observe(self, o, context):
        """Return (role, candidate team, reason, detailed evidence)."""
        c = self.config
        tid, frame = int(o["track_id"]), int(o["frame_id"])
        observation_gap = frame - self.last_seen.get(tid, -10**9)
        self.last_seen[tid] = frame
        bbox = o["bbox"]
        foot = float(o["foot"])
        playline = o["playline"]
        appearance = o["appearance"]
        instant = o["instant"]
        cohort = context["cohort_foot_p90"]
        # A near-bottom box far deeper than the current player cohort is
        # foreground evidence. No single footpoint or mask value suffices.
        foreground = (context["cohort_count"] >= c["min_cohort_count"] and
                      cohort is not None and foot >= c["foreground_min_foot"] and
                      foot > cohort + c["foreground_cohort_gap"])
        foreground_unknown = (context["cohort_count"] < c["min_cohort_count"] and
                              foot >= c["foreground_without_cohort_foot"])
        center = ((bbox[0] + bbox[2]) / (2 * o["width"]),
                  (bbox[1] + bbox[3]) / (2 * o["height"]))
        old = self.previous_center.get(tid)
        residual = 0.0 if old is None else ((center[0] - old[0] - context["camera_dx"]) ** 2 +
                                            (center[1] - old[1] - context["camera_dy"]) ** 2) ** .5
        self.previous_center[tid] = center
        supported = o["court_reason"] in ("court", "boundary_grace")
        # An absent/tiny cohort supplies no positive evidence. A previously
        # confirmed player may instead use the short grace below.
        good_spatial = (supported and not foreground and not foreground_unknown and
                        context["cohort_count"] >= c["min_cohort_count"] and
                        cohort is not None and foot <= cohort + c["player_cohort_margin"])
        last_foreground = self.last_persistent_foreground.get(tid, -10**9)
        foreground_lock = (frame - last_foreground <= c["foreground_memory_frames"] and
                           self.foreground_recovery[tid] < c["foreground_recovery_frames"])
        # In a sparse camera view, repeated interior-court evidence plus a
        # jersey candidate can establish a new player. The court mask alone
        # and a single clothing vote never suffice.
        interior_cue = (supported and bool(o.get("interior", False)) and
                        o["raw_team"] in ("A", "B") and instant == "active_player" and
                        not foreground and not foreground_unknown and not foreground_lock)
        self.interior_player_streak[tid] = self.interior_player_streak[tid] + 1 if interior_cue else 0
        sustained_interior = self.interior_player_streak[tid] >= c["low_cohort_interior_min_frames"]
        spatial_player = (good_spatial or sustained_interior) and not foreground_lock
        team_uniform = (appearance.get("purple_fraction", 0) >= c["ref_veto_purple_shirt"] or
                        appearance.get("lower_purple", 0) >= c["ref_veto_purple_lower"] or
                        appearance.get("lower_yellow", 0) >= c["ref_veto_yellow_lower"] or
                        appearance.get("strong_team_evidence") in ("A", "B"))
        stripe = (appearance["center_stripe_transitions"] >= 1 and
                  appearance["center_stripe_coverage"] >= c["ref_min_coverage"] and
                  appearance["center_stripe_dark"] >= c["ref_min_dark"] and
                  appearance["center_stripe_light"] >= c["ref_min_light"] and
                  appearance["lower_dark"] >= c["ref_min_lower_dark"] and
                  not team_uniform)
        # Duty is independent of stripe appearance. A seated table-side crop
        # needs both position and some stripe structure, not merely a shirt.
        table_group = (context["table_group_count"] >= c["table_min_group"] and
                       context["table_x_min"] is not None and
                       context["table_x_min"] - c["table_group_margin"] <= center[0] <=
                       context["table_x_max"] + c["table_group_margin"])
        table_position = (table_group and playline is not None and
                          foot < playline - c["table_playline_gap"] and
                          (bbox[3] - bbox[1]) / o["height"] < c["table_max_height"] and
                          o["court_reason"] == "outside_court")
        table_stripe = (appearance["center_stripe_transitions"] >= 1 and
                        appearance["center_stripe_coverage"] >= c["table_min_coverage"] and
                        appearance["center_stripe_light"] >= c["table_min_light"])
        table_cue = table_position and table_stripe and residual <= c["table_max_residual"]
        referee_cue = stripe and supported and not foreground and not table_position
        history = self.history[tid]
        foreground_votes = sum(x[4] for x in list(history)[-(c["foreground_window"]-1):]) + bool(foreground)
        if foreground_votes >= c["revoke_foreground_votes"]:
            self.last_persistent_foreground[tid] = frame
            foreground_lock = True
            self.foreground_recovery[tid] = 0
            spatial_player = False
        if good_spatial:
            self.foreground_recovery[tid] += 1
        else:
            self.foreground_recovery[tid] = 0
        if (good_spatial or sustained_interior) and not foreground_lock:
            self.reliable_player_observations[tid] += 1
            self.last_reliable_player_observation[tid] = frame
        prior_reliable = self.last_reliable_player_observation.get(tid, -10**9)
        established_player = (self.reliable_player_observations[tid] >= c["established_player_min_good_frames"] and
                              frame - prior_reliable <= c["established_player_grace_frames"] and
                              supported and not foreground and not foreground_unknown and
                              not foreground_lock)
        spatial_player = spatial_player or established_player
        player_cue = instant == "active_player" and spatial_player
        history.append((frame, player_cue, referee_cue, table_cue, foreground,
                        spatial_player, residual))
        recent = list(history)
        player_votes = sum(x[1] for x in recent)
        referee_votes = sum(x[2] for x in recent)
        table_votes = sum(x[3] for x in recent)
        prev_active = self.last_active.get(tid, -10**9)
        active_grace = (frame - prev_active <= c["active_grace_frames"] and
                        not foreground and not foreground_unknown and
                        not foreground_lock and
                        foreground_votes < c["revoke_foreground_votes"] and
                        instant in ("uncertain", "active_player"))
        # V5.6: a recently confirmed, repeatedly observed player may keep the
        # active role through brief weak appearance/participation cues. This
        # never establishes a new player, bypasses a foreground/table veto,
        # or survives a substantial tracking gap or lost court support.
        participation_grace = (
            frame - prev_active <= c["v56_participation_grace_frames"] and
            observation_gap <= c["v56_max_observation_gap_frames"] and
            self.reliable_player_observations[tid] >= c["established_player_min_good_frames"] and
            supported and not foreground and not foreground_unknown and
            not foreground_lock and not table_position and
            not stripe and foreground_votes < c["revoke_foreground_votes"])
        if foreground_votes >= c["revoke_foreground_votes"]:
            role, reason = "nonparticipant", "persistent_foreground_relative_to_cohort"
        elif table_cue and table_votes >= c["table_min_votes"]:
            role, reason = "table_official", "repeated_seated_table_stripe_evidence"
        elif referee_cue and referee_votes >= c["ref_min_votes"]:
            role, reason = "referee", "repeated_stripes_with_court_duty"
        elif player_cue and player_votes >= c["player_min_votes"]:
            role, reason = "active_player", "repeated_spatial_player_evidence"
        elif active_grace:
            role, reason = "active_player", "recent_confirmed_player_grace"
        elif participation_grace:
            role, reason = "active_player", "v56_confirmed_player_participation_grace"
        elif instant == "nonparticipant" and not supported:
            role, reason = "nonparticipant", "outside_play_without_role_evidence"
        else:
            role, reason = "uncertain", "participation_not_confirmed"
        if role == "active_player" and reason not in (
                "recent_confirmed_player_grace", "v56_confirmed_player_participation_grace"):
            self.last_active[tid] = frame
        candidate = o["raw_team"] if role == "active_player" and player_cue else None
        detail = {"foreground_candidate": foreground, "foreground_votes": foreground_votes,
                  "foreground_without_cohort": foreground_unknown,
                  "cohort_count": context["cohort_count"], "cohort_foot_p90": cohort,
                  "camera_dx": context["camera_dx"], "camera_dy": context["camera_dy"],
                  "motion_residual": round(residual, 5), "spatial_player": spatial_player,
                  "player_votes": player_votes, "referee_votes": referee_votes,
                  "table_votes": table_votes, "table_position": table_position,
                  "table_stripe": table_stripe, "table_group_count": context["table_group_count"],
                  "referee_appearance": stripe,
                  "team_uniform_veto": team_uniform,
                  "cohort_reliable": context["cohort_count"] >= c["min_cohort_count"],
                  "positive_player_spatial": good_spatial,
                  "sustained_interior_player_evidence": sustained_interior,
                  "interior_player_streak": self.interior_player_streak[tid],
                  "established_player_persistence": established_player,
                  "reliable_player_observations": self.reliable_player_observations[tid],
                  "foreground_memory_active": foreground_lock,
                  "foreground_recovery_count": self.foreground_recovery[tid],
                  "participation_grace_v56": participation_grace,
                  "observation_gap_v56": observation_gap,
                  "referee_duty": supported and not foreground,
                  "role_change_reason": reason}
        return role, candidate, reason, detail


def snapshot_from_evidence(appearance, court_reason, foot, playline, config):
    """Reuse V5.1's single-frame cue for saved-evidence replay and runtime."""
    return classify_snapshot(appearance, court_reason, foot, playline, config)
