"""Reversible, per-track team memory for confirmed active basketball players."""

from collections import Counter, defaultdict, deque


class TeamVotesV55:
    def __init__(self, classification_config, memory_config):
        self.classification = classification_config
        self.memory = memory_config
        self.votes = defaultdict(lambda: deque(maxlen=classification_config["vote_window"]))
        self.state = {}
        self.last_seen = {}
        self.last_active = {}
        self.last_team_evidence = {}
        self.opposite_streak = defaultdict(int)
        self.last_decision = {}

    def _clear(self, tid):
        self.votes.pop(tid, None)
        self.state.pop(tid, None)
        self.last_active.pop(tid, None)
        self.last_team_evidence.pop(tid, None)
        self.opposite_streak.pop(tid, None)

    def observe(self, track_id, frame_id, role, raw_vote):
        tid, frame = int(track_id), int(frame_id)
        previous = self.last_seen.get(tid)
        gap_reset = previous is not None and frame - previous > self.memory["max_track_gap_frames"]
        if gap_reset:
            self._clear(tid)
        self.last_seen[tid] = frame
        reason = "unconfirmed"
        if role in ("nonparticipant", "referee", "table_official"):
            self._clear(tid)
            reason = "confirmed_nonplayer_reset"
            output = None
        elif role != "active_player":
            active = self.last_active.get(tid)
            if active is not None and frame - active > self.memory["uncertain_role_grace_frames"]:
                self._clear(tid)
                reason = "uncertain_role_expired"
            else:
                reason = "team_hidden_during_uncertain_role"
            output = None
        else:
            self.last_active[tid] = frame
            if raw_vote in ("A", "B"):
                self.votes[tid].append((frame, raw_vote))
                self.last_team_evidence[tid] = frame
            recent = [vote for at, vote in self.votes[tid]
                      if frame - at <= self.classification["vote_max_age_frames"]]
            current = self.state.get(tid)
            if current is None:
                if recent:
                    leader, count = Counter(recent).most_common(1)[0]
                    if (count >= self.classification["min_votes"] and
                            count / len(recent) >= self.classification["vote_fraction"]):
                        self.state[tid] = leader
                        current = leader
                        reason = "team_confirmed_by_votes"
                    else:
                        reason = "awaiting_team_votes"
                else:
                    reason = "no_team_evidence"
            else:
                if raw_vote in ("A", "B") and raw_vote != current:
                    self.opposite_streak[tid] += 1
                    if self.opposite_streak[tid] >= self.memory["switch_consecutive_votes"]:
                        current = raw_vote
                        self.state[tid] = current
                        self.opposite_streak[tid] = 0
                        reason = "sustained_opposite_jersey_switch"
                    else:
                        reason = "contradictory_jersey_pending"
                elif raw_vote == current:
                    self.opposite_streak[tid] = 0
                    reason = "matching_jersey_support"
                else:
                    self.opposite_streak[tid] = 0
                    reason = "weak_jersey_team_memory"
                last_evidence = self.last_team_evidence.get(tid)
                if (last_evidence is None or
                        frame - last_evidence > self.memory["active_no_jersey_expiry_frames"]):
                    self._clear(tid)
                    current = None
                    reason = "team_evidence_expired"
            output = current
        recent = [vote for at, vote in self.votes.get(tid, ())
                  if frame - at <= self.classification["vote_max_age_frames"]]
        ratio = (round(Counter(recent)[output] / len(recent), 3)
                 if output is not None and recent else 0.0)
        self.last_decision[tid] = {
            "team_memory": self.state.get(tid),
            "team_change_reason": "track_gap_reset;" + reason if gap_reset else reason,
            "team_opposite_streak": self.opposite_streak.get(tid, 0),
            "team_last_active_frame": self.last_active.get(tid),
        }
        return output, ratio
