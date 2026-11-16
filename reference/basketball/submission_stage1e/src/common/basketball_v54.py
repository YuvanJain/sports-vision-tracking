"""V5.4 reversible referee state with separate appearance and duty memories."""

from collections import defaultdict, deque

from .basketball_v54_participation import RoleClassifierV54Participation


class RoleClassifierV54(RoleClassifierV54Participation):
    def __init__(self, participant_config, referee_config):
        super().__init__(participant_config)
        self.ref_config = referee_config
        self.ref_state = {}
        self.appearance_history = defaultdict(lambda: deque(maxlen=referee_config["duty_window_frames"]))
        self.duty_history = defaultdict(lambda: deque(maxlen=referee_config["duty_window_frames"]))

    def observe(self, observation, context):
        tid, frame = int(observation["track_id"]), int(observation["frame_id"])
        c = self.ref_config
        prior = self.ref_state.get(tid)
        if prior and frame - prior["last_seen"] > c["max_track_gap_frames"]:
            prior = None
            self.appearance_history.pop(tid, None)
            self.duty_history.pop(tid, None)
        if prior is None:
            prior = {"confirmed": False, "last_seen": frame, "last_strong": None,
                     "strong_frames": [], "last_player_cue": None,
                     "player_streak": 0, "duty_loss_streak": 0, "foreground_streak": 0}
        prior["last_seen"] = frame
        role, candidate, reason, detail = super().observe(observation, context)
        supported = observation["court_reason"] in ("court", "boundary_grace")
        table = detail["table_position"]
        foreground = detail["foreground_candidate"] or detail["foreground_without_cohort"]
        prior["foreground_streak"] = prior["foreground_streak"] + 1 if foreground else 0
        # Only a previously confirmed official with repeated recent appearance
        # can temporarily cross the camera-relative foreground boundary.
        recent_stripes = sum(value for _, value in self.appearance_history[tid])
        foreground_exception = (foreground and prior["confirmed"] and supported and not table and
                                recent_stripes >= c["foreground_min_recent_stripes"] and
                                prior["foreground_streak"] <= c["foreground_ref_grace_frames"] and
                                observation["foot"] < c["foreground_ref_max_foot"] and
                                not detail["team_uniform_veto"])
        duty = supported and not table and (not foreground or foreground_exception)
        strong = detail["referee_appearance"] and duty and not detail["team_uniform_veto"]
        self.appearance_history[tid].append((frame, bool(detail["referee_appearance"])))
        self.duty_history[tid].append((frame, bool(duty)))
        prior["strong_frames"] = [f for f in prior["strong_frames"]
                                  if frame - f < self.config["evidence_window"]]
        if strong:
            prior["last_strong"] = frame
            if not prior["confirmed"]:
                prior["strong_frames"].append(frame)
        if not prior["confirmed"] and detail["spatial_player"] and observation["instant"] == "active_player":
            prior["last_player_cue"] = frame
        recent_player = (prior["last_player_cue"] is not None and
                         frame - prior["last_player_cue"] <= c["new_referee_player_block_frames"])
        contradiction = (detail["spatial_player"] and observation["instant"] == "active_player" and
                         detail["team_uniform_veto"] and not strong)
        prior["player_streak"] = prior["player_streak"] + 1 if contradiction else 0
        prior["duty_loss_streak"] = 0 if duty else prior["duty_loss_streak"] + 1
        duty_votes = sum(value for _, value in self.duty_history[tid])
        appearance_votes = sum(value for _, value in self.appearance_history[tid])
        age = frame - prior["last_strong"] if prior["last_strong"] is not None else None
        action = "unconfirmed"

        reliable_nonparticipant = table or (detail["foreground_votes"] >=
                                           self.config["revoke_foreground_votes"] and
                                           not foreground_exception)
        if reliable_nonparticipant or prior["duty_loss_streak"] >= c["duty_loss_revoke_frames"]:
            prior["confirmed"] = False
            prior["strong_frames"] = []
            action = "nonparticipant_or_duty_revoke"
            if role == "referee":
                role, candidate, reason = "uncertain", None, "referee_duty_contradicted"
        elif prior["player_streak"] >= c["player_override_consecutive"]:
            prior["confirmed"] = False
            prior["strong_frames"] = []
            action = "sustained_player_contradiction"
        elif not prior["confirmed"] and strong and not recent_player and len(prior["strong_frames"]) >= self.config["ref_min_votes"]:
            prior["confirmed"] = True
            role, candidate, reason = "referee", None, "repeated_referee_appearance_and_duty"
            action = "new_referee_confirmed"
        elif prior["confirmed"] and duty and strong:
            role, candidate, reason = "referee", None, "confirmed_referee_appearance"
            action = "strong_referee_evidence"
        elif (prior["confirmed"] and duty and age is not None and
              age <= c["appearance_memory_frames"] and
              duty_votes >= min(c["min_duty_votes"], len(self.duty_history[tid]))):
            role, candidate, reason = "referee", None, "appearance_memory_with_continued_duty"
            action = "referee_memory_hold"
        elif prior["confirmed"] and age is not None and age >= c["revoke_no_cue_frames"]:
            prior["confirmed"] = False
            prior["strong_frames"] = []
            action = "appearance_memory_expired"
            if role == "referee":
                role, candidate, reason = "uncertain", None, "referee_evidence_expired"
        elif prior["confirmed"]:
            role, candidate, reason = "uncertain", None, "referee_evidence_or_duty_insufficient"
            action = "confirmed_but_not_displayed"
        elif role == "referee":
            role, candidate, reason = "uncertain", None, "referee_reconfirmation_pending"
            action = "awaiting_referee_evidence"

        self.ref_state[tid] = prior
        detail.update({"ref_confirmed": prior["confirmed"],
                       "ref_last_strong_frame": prior["last_strong"],
                       "ref_weak_age_frames": age,
                       "ref_player_contradiction_streak": prior["player_streak"],
                       "ref_recent_player_block": recent_player and not prior["confirmed"],
                       "ref_new_strong_votes": len(prior["strong_frames"]),
                       "ref_state_action": action,
                       "ref_appearance_votes": appearance_votes,
                       "ref_duty_votes": duty_votes,
                       "ref_duty_effective": duty,
                       "ref_duty_loss_streak": prior["duty_loss_streak"],
                       "ref_foreground_exception": foreground_exception,
                       "role_change_reason": reason})
        return role, candidate, reason, detail
