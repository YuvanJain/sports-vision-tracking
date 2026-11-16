"""Reversible referee hysteresis layered on V5.2 participation decisions.

This class consumes existing per-track appearance and court observations. It
does not alter person detection, tracking, team-color definitions or ball logic.
"""

from .basketball_v52 import RoleClassifierV52


class RoleClassifierV53(RoleClassifierV52):
    def __init__(self, participant_config, hysteresis_config):
        super().__init__(participant_config)
        self.hysteresis = hysteresis_config
        self.referee_state = {}

    def observe(self, observation, context):
        tid = int(observation["track_id"])
        frame = int(observation["frame_id"])
        prior = self.referee_state.get(tid)
        if prior is not None and frame - prior["last_seen"] > self.hysteresis["max_track_gap_frames"]:
            # A stale ByteTrack ID must earn referee status again after a gap.
            # Keep V5.2's player/team history unchanged.
            self.referee_state.pop(tid, None)
            prior = None

        base_role, candidate, reason, detail = super().observe(observation, context)
        strong = (detail["referee_appearance"] and detail["referee_duty"] and
                  not detail["table_position"] and
                  detail["foreground_votes"] < self.config["revoke_foreground_votes"])
        blocked = (observation["court_reason"] not in ("court", "boundary_grace") or
                   detail["foreground_candidate"] or detail["foreground_without_cohort"] or
                   detail["foreground_votes"] >= self.config["revoke_foreground_votes"] or
                   detail["table_position"])
        contradiction = (detail["spatial_player"] and
                         observation["instant"] == "active_player" and
                         detail["team_uniform_veto"] and not strong)
        action = "unconfirmed"

        if prior is None:
            prior = {"confirmed": False, "last_strong": None,
                     "last_seen": frame, "player_streak": 0,
                     "last_player_cue": None, "new_strong_frames": []}
        prior["last_seen"] = frame
        prior["new_strong_frames"] = [f for f in prior["new_strong_frames"]
                                      if frame - f < self.config["evidence_window"]]
        if not prior["confirmed"] and (detail["spatial_player"] and
                                        observation["instant"] == "active_player"):
            prior["last_player_cue"] = frame
        recent_player = (prior["last_player_cue"] is not None and
                         frame - prior["last_player_cue"] <=
                         self.hysteresis["new_referee_player_block_frames"])

        if blocked:
            # Never carry on-court duty into the crowd or scorer table.
            prior["confirmed"] = False
            prior["player_streak"] = 0
            prior["new_strong_frames"] = []
            action = "duty_or_foreground_rejected"
            if base_role == "referee":
                base_role, candidate, reason = "uncertain", None, "referee_duty_contradicted"
        elif strong:
            prior["last_strong"] = frame
            prior["player_streak"] = 0
            if not prior["confirmed"]:
                prior["new_strong_frames"].append(frame)
            if base_role == "referee" and not prior["confirmed"] and recent_player:
                base_role, candidate, reason = "uncertain", None, "recent_player_evidence_blocks_referee"
                action = "new_referee_confirmation_blocked"
            elif (base_role == "referee" and not prior["confirmed"] and
                  len(prior["new_strong_frames"]) < self.config["ref_min_votes"]):
                base_role, candidate, reason = "uncertain", None, "referee_reconfirmation_pending"
                action = "awaiting_new_referee_evidence"
            elif base_role == "referee" or prior["confirmed"]:
                prior["confirmed"] = True
                base_role, candidate, reason = "referee", None, "confirmed_referee_stripes"
                action = "strong_referee_evidence"
        elif prior["confirmed"]:
            prior["player_streak"] = prior["player_streak"] + 1 if contradiction else 0
            age = frame - prior["last_strong"]
            if prior["player_streak"] >= self.hysteresis["player_override_consecutive"]:
                prior["confirmed"] = False
                prior["new_strong_frames"] = []
                action = "sustained_player_contradiction"
            elif age <= self.hysteresis["hold_weak_frames"]:
                base_role, candidate, reason = "referee", None, "recent_confirmed_referee_grace"
                action = "weak_stripe_grace"
            elif age < self.hysteresis["revoke_no_cue_frames"]:
                base_role, candidate, reason = "uncertain", None, "referee_evidence_stale"
                action = "stale_referee_evidence"
            else:
                prior["confirmed"] = False
                prior["new_strong_frames"] = []
                action = "persistent_absence_of_referee_evidence"

        self.referee_state[tid] = prior
        detail.update({"ref_confirmed": prior["confirmed"],
                       "ref_last_strong_frame": prior["last_strong"],
                       "ref_weak_age_frames": (frame - prior["last_strong"]
                                               if prior["last_strong"] is not None else None),
                       "ref_player_contradiction_streak": prior["player_streak"],
                       "ref_recent_player_block": recent_player and not prior["confirmed"],
                       "ref_new_strong_votes": len(prior["new_strong_frames"]),
                       "ref_state_action": action,
                       "role_change_reason": reason})
        return base_role, candidate, reason, detail
