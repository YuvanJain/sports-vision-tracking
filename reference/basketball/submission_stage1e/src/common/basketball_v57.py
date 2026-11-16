"""Conservative close-up player extension; no detection or tracker changes.

Saved box discontinuities are cut *suspicions*, not pixel-based cut detection.
Never establish participation from clothing alone or transfer identity across IDs.
"""
from collections import deque
from math import hypot
from .basketball_closeup_referee import RoleClassifierCloseup
from .basketball_v55_team import TeamVotesV55

FIELDS_V57 = ('v57_action', 'v57_prior_court_observations', 'v57_uniform_streak',
              'v57_scene_epoch', 'v57_cut_suspected', 'v57_reset_team',
              'v57_anchor_age', 'v57_uniform_team', 'v57_baseline_role', 'v57_baseline_reason')


class PlayerCloseupV57(RoleClassifierCloseup):
    def __init__(self, config):
        super().__init__(config['participant_v52'], config['referee_memory_v54'],
                         config['closeup_referee_v56_final'])
        self.c = config['closeup_player_v57']
        self.uniform_config = config['participant']
        self.memory = {}
        self.boxes = {}
        self.epoch = 0
        self.cut = False
        self.last_gameplay = -1000000

    @staticmethod
    def geometry(o):
        x1, y1, x2, y2 = o['bbox']; w, h = o['width'], o['height']
        return ((x1+x2)/2/w, (y1+y2)/2/h, (y2-y1)/h)

    def begin_frame(self, observations, context):
        current = {int(o['track_id']): self.geometry(o) for o in observations}
        shared = current.keys() & self.boxes.keys()
        jumps = sum(self.discontinuous(self.boxes[t], current[t]) for t in shared)
        self.cut = bool(context.get('scene_cut', False) or
                        (len(shared) >= self.c['cut_min_shared'] and
                         jumps / len(shared) >= self.c['cut_jump_fraction']))
        if self.cut:
            self.epoch += 1
            self.last_gameplay = -1000000
            # Invalidate spatial/player evidence; preserve referee appearance memory
            # and conservative nonparticipant/table exclusions.
            for name in ('previous_center', 'last_active', 'last_team',
                         'reliable_player_observations', 'last_reliable_player_observation',
                         'interior_player_streak'):
                getattr(self, name).clear()
            for tid, history in self.history.items():
                self.history[tid] = deque(((f, False, r, t, fg, False, 0.)
                                          for f, p, r, t, fg, s, m in history),
                                         maxlen=history.maxlen)
        if observations and not self.cut and context['cohort_count'] >= self.c['scene_min_cohort']:
            self.last_gameplay = observations[0]['frame_id']
        self.boxes = current

    def discontinuous(self, before, after):
        return (hypot(before[0]-after[0], before[1]-after[1]) > self.c['max_center_jump'] or
                max(before[2], after[2]) / max(.001, min(before[2], after[2])) > self.c['max_height_ratio'])

    def uniform(self, a):
        c = self.uniform_config
        if not a.get('crop_valid', False): return None
        b = (a.get('strong_team_evidence') == 'B' and a.get('purple_fraction', 0) >= c['b_shirt_min'] and
             max(a.get('lower_purple', 0), a.get('lower_yellow', 0)) >= c['b_lower_min'])
        white = (a.get('strong_team_evidence') == 'A' and a.get('white_fraction', 0) >= c['a_shirt_min'] and
                 a.get('lower_white', 0) >= c['a_lower_min'])
        return 'B' if b and not white else 'A' if white and not b else None

    def observe(self, o, context):
        tid, frame = int(o['track_id']), int(o['frame_id']); geom = self.geometry(o)
        old = self.memory.get(tid)
        was_extended = bool(old and old.get('extended', False))
        reset = old is not None and (old['epoch'] != self.epoch or
                frame-old['seen'] > self.c['max_gap_frames'] or self.discontinuous(old['geom'], geom))
        if old is None or reset:
            old = {'epoch': self.epoch, 'seen': frame, 'geom': geom, 'court': deque(),
                   'anchor': None, 'anchor_team': None, 'streak': 0, 'team': None,
                   'contradictions': 0, 'extended': False}
        role, team, reason, detail = super().observe(o, context)
        baseline_role, baseline_reason, baseline_team = role, reason, team
        # A baseline grace prediction is not a fresh participation observation.
        # Re-evaluate it through the bounded close-up route without refreshing
        # the trusted anchor or letting contradictions bypass abstention.
        if role == 'active_player' and o['court_reason'] == 'mask_uncertain':
            role, team, reason = 'uncertain', None, 'closeup_requires_player_memory'
        kit = self.uniform(o['appearance'])
        old['seen'], old['geom'] = frame, geom
        while old['court'] and frame-old['court'][0] > self.c['court_history_frames']:
            old['court'].popleft()
        if detail.get('positive_player_spatial') and o['instant'] == 'active_player' and kit:
            old['court'].append(frame)
        veto = (role in ('referee', 'nonparticipant', 'table_official') or
                detail.get('referee_appearance') or detail.get('foreground_candidate') or
                detail.get('foreground_without_cohort') or detail.get('foreground_memory_active') or
                detail.get('table_position') or context.get('table_group_count', 0) >= 2)
        uncertain_court = o['court_reason'] == 'mask_uncertain' and o.get('court_mask_status') == 'uncertain'
        geometry = (geom[2] >= self.c['min_height_fraction'] and
                    self.c['min_center_x'] <= geom[0] <= self.c['max_center_x'] and
                    o['foot'] < self.c['max_foot_fraction'])
        action = 'baseline'
        if veto:
            old.update(anchor=None, anchor_team=None, streak=0, team=None, contradictions=0, extended=False)
            old['court'].clear(); action = 'role_or_foreground_veto'
        elif role == 'active_player' and detail.get('positive_player_spatial'):
            # Only the unchanged baseline can refresh the bounded memory anchor.
            old['anchor'] = frame
            if kit: old['anchor_team'] = kit
            old['streak'] = 0
            old['extended'] = False
        elif uncertain_court and geometry and not reset and not self.cut:
            old['streak'] = old['streak'] + 1 if kit and kit == old['team'] else int(bool(kit))
            old['team'] = kit
            opposite = bool(kit and old['anchor_team'] and kit != old['anchor_team'])
            old['contradictions'] = old['contradictions'] + 1 if opposite else 0
            if old['contradictions'] >= self.c['contradiction_frames']:
                old['anchor'] = None; old['court'].clear()
            anchored = (old['anchor'] is not None and
                        frame-old['anchor'] <= self.c['hold_frames'])
            supported = (len(old['court']) >= self.c['min_court_observations'] and
                         frame-self.last_gameplay <= self.c['scene_history_frames'])
            if opposite:
                action = 'contradictory_uniform_abstain'
            elif anchored:
                role, team, action = 'active_player', kit, 'bounded_confirmed_player_hold'
            elif supported and old['streak'] >= self.c['uniform_confirm_frames']:
                role, team, action = 'active_player', kit, 'prior_court_and_repeated_full_uniform'
            else:
                action = 'closeup_insufficient_participation_history'
            if role == 'active_player':
                reason = action
                old['extended'] = True
        else:
            old['streak'] = 0; action = 'geometry_or_continuity_ineligible'
        # This is an extension, not a replacement for existing safe gameplay
        # grace. A smaller box need not satisfy the new close-up geometry to
        # retain the baseline's own bounded result. Do not refresh our anchor.
        if (baseline_role == 'active_player' and role == 'uncertain' and
                not veto and not reset and not self.cut and action in (
                    'geometry_or_continuity_ineligible', 'closeup_insufficient_participation_history')):
            role, team, reason = baseline_role, baseline_team, baseline_reason
            action = 'unchanged_baseline_grace'
        self.memory[tid] = old
        detail.update(dict(zip(FIELDS_V57, (action, len(old['court']), old['streak'], self.epoch,
                      self.cut, (reset and was_extended) or self.cut,
                      None if old['anchor'] is None else frame-old['anchor'], kit,
                      baseline_role, baseline_reason))))
        detail['role_change_reason'] = reason
        return role, team, reason, detail


class TeamsV57(TeamVotesV55):
    def observe_with_detail(self, tid, frame, role, vote, detail):
        if detail['v57_reset_team']: self._clear(tid)
        return self.observe(tid, frame, role, vote)
