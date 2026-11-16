"""Conservative referee evidence when a broadcast close-up hides the court.

This is a separate V5.6 final candidate. It never changes detection, tracking,
player/team voting, or the V5.4 court-visible referee rule.
"""

from collections import defaultdict, deque

from .basketball_v54 import RoleClassifierV54


class RoleClassifierCloseup(RoleClassifierV54):
    def __init__(self, participant_config, referee_config, closeup_config):
        super().__init__(participant_config, referee_config)
        self.closeup = closeup_config
        self.closeup_history = defaultdict(lambda: deque(maxlen=closeup_config["appearance_window_frames"]))
        self.closeup_state = {}

    def observe(self, observation, context):
        role, team, reason, detail = super().observe(observation, context)
        c = self.closeup
        tid, frame = int(observation["track_id"]), int(observation["frame_id"])
        old = self.closeup_state.get(tid)
        if old is None or frame - old["last_seen"] > c["max_observation_gap_frames"]:
            old = {"confirmed": False, "last_seen": frame,
                   "last_strong": None, "team_veto_streak": 0}
            self.closeup_history.pop(tid, None)
        old["last_seen"] = frame

        box = observation["bbox"]
        height = (box[3] - box[1]) / observation["height"]
        foot = observation["foot"]
        mask_unreliable = (observation.get("court_mask_status") == "uncertain" and
                           observation["court_reason"] == "mask_uncertain")
        geometry = (height >= c["min_bbox_height_fraction"] and
                    c["min_foot_fraction"] <= foot <= c["max_foot_fraction"])
        recent_player = (frame - self.last_active.get(tid, -10**9) <=
                         c["recent_player_block_frames"])
        hard_veto = (role in ("active_player", "nonparticipant", "table_official") or
                     detail["foreground_candidate"] or detail["foreground_memory_active"] or
                     detail["table_position"] or
                     detail["table_group_count"] >= c["table_group_veto_count"] or
                     recent_player)
        appearance = observation["appearance"]
        strong = (mask_unreliable and geometry and not hard_veto and
                  not detail["team_uniform_veto"] and detail["referee_appearance"] and
                  appearance["center_stripe_transitions"] >= c["min_center_stripe_transitions"] and
                  appearance["center_stripe_coverage"] >= c["min_center_stripe_coverage"])

        if not mask_unreliable or not geometry or hard_veto:
            old["confirmed"] = False
            old["last_strong"] = None
            old["team_veto_streak"] = 0
            self.closeup_history.pop(tid, None)
            action = "closeup_ineligible"
        else:
            old["team_veto_streak"] = (old["team_veto_streak"] + 1
                                       if detail["team_uniform_veto"] else 0)
            self.closeup_history[tid].append(bool(strong))
            votes = sum(self.closeup_history[tid])
            if old["team_veto_streak"] >= c["team_veto_revoke_frames"]:
                old["confirmed"] = False
                self.closeup_history.pop(tid, None)
                action = "closeup_team_contradiction"
            elif strong and votes >= c["min_strong_votes"]:
                old["confirmed"] = True
                old["last_strong"] = frame
                action = "closeup_repeated_stripes_confirmed"
            elif old["confirmed"] and old["last_strong"] is not None and (
                    frame - old["last_strong"] <= c["weak_stripe_hold_frames"]):
                action = "closeup_bounded_appearance_hold"
            else:
                old["confirmed"] = False
                action = "closeup_awaiting_stripes"

        if old["confirmed"] and role == "uncertain":
            role, team, reason = "referee", None, action
            detail["ref_confirmed"] = True
            detail["ref_state_action"] = action
            detail["role_change_reason"] = reason
        self.closeup_state[tid] = old
        detail.update({"closeup_mask_unreliable": mask_unreliable,
                       "closeup_geometry": geometry,
                       "closeup_strong_stripe": strong,
                       "closeup_stripe_votes": sum(self.closeup_history[tid]),
                       "closeup_confirmed": old["confirmed"],
                       "closeup_last_strong_frame": old["last_strong"],
                       "closeup_team_veto_streak": old["team_veto_streak"],
                       "closeup_action": action})
        return role, team, reason, detail
