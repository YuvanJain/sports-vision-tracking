"""Stage 1B: scene-scoped role memory and evidence-gated new player route."""
from collections import deque
from .basketball_v57 import PlayerCloseupV57, TeamsV57, FIELDS_V57

FIELDS_1B = ('scene_epoch', 'scene_cut_confirmed', 'scene_quarantine',
    'independent_match_scene', 'independent_person_support', 'independent_uniform_streak',
    'stage1b_action', 'identity_scope', 'stage1b_reset_team', 'independent_scene_recent_votes', 'independent_scene_effective')


class PlayerStage1B(PlayerCloseupV57):
    def __init__(self, config):
        super().__init__(config)
        self.full_config = config
        self.new_config = config['player_stage1b']
        self.new_history = {}
        self.scene_epoch = 0
        self.reset_frame = False
        self.pending_reset = False
        self.scene_last_seen = {}
        self.scene_support = deque()

    def begin_frame(self, observations, context):
        self.reset_frame = bool(context.get('scene_cut') or context.get('scene_quarantine'))
        if self.reset_frame:
            epoch = self.scene_epoch + 1
            # Reset all role evidence, including old table/referee/player
            # associations; current-scene referee rules remain byte-identical.
            self.__init__(self.full_config)
            self.scene_epoch = epoch
            self.reset_frame = True
        # Stage 1's box-motion cut proxy is not used here. Only pixel evidence
        # may establish a scene change; a pan or moving athlete is not a cut.
        self.boxes = {}
        clean_context = dict(context, scene_cut=False)
        super().begin_frame(observations, clean_context)
        self.cut = bool(context.get('scene_cut'))
        if observations:
            frame=observations[0]['frame_id']
            if context.get('independent_match_scene') and not self.reset_frame:self.scene_support.append(frame)
            while self.scene_support and frame-self.scene_support[0]>self.new_config['scene_window_frames']:
                self.scene_support.popleft()

    def observe(self, o, context):
        tid, frame = int(o['track_id']), int(o['frame_id'])
        previous = self.scene_last_seen.get(tid)
        gap = previous is not None and frame-previous > self.new_config['max_observation_gap']
        self.scene_last_seen[tid] = frame
        role, team, reason, detail = super().observe(o, context)
        kit = self.uniform(o['appearance'])
        old = self.new_history.get(tid, dict(streak=0, team=None, seen=frame, confirmed=None))
        was_new_player = old['confirmed'] is not None
        if gap or self.reset_frame:
            old = dict(streak=0, team=None, seen=frame, confirmed=None)
        visual = o.get('scene_person', {})
        observed_scene = bool(context.get('independent_match_scene'))
        # A few missing line observations within the SAME view must not erase
        # accumulated independent evidence. Missing pixels cannot extend it.
        scene = bool(len(self.scene_support)>=self.new_config['min_scene_observations'] and
            (observed_scene or (context.get('scene_visual_available') and self.scene_support and
             frame-self.scene_support[-1]<=self.new_config['scene_hold_frames'])))
        spatial = bool(visual.get('independent_person_support'))
        # Fixed shin/short sampling can miss shorts in a low-angle full-body
        # view. The alternative waist crop is evidence for THIS route only;
        # original appearance, referee vetoes and team thresholds are unchanged.
        if kit is None and scene and spatial:
            a=o['appearance']; u=self.uniform_config
            if (a.get('strong_team_evidence')=='A' and a.get('white_fraction',0)>=u['a_shirt_min']
                    and visual.get('waist_white',0)>=u['a_lower_min']): kit='A'
            elif (a.get('strong_team_evidence')=='B' and a.get('purple_fraction',0)>=u['b_shirt_min']
                    and max(visual.get('waist_purple',0),visual.get('waist_yellow',0))>=u['b_lower_min']): kit='B'
        veto = (role in ('referee','table_official','nonparticipant') or
                detail.get('referee_appearance') or detail.get('foreground_candidate') or
                (detail.get('foreground_without_cohort') and not (scene and spatial)) or detail.get('foreground_memory_active') or
                detail.get('table_position') or context.get('table_group_count',0) >= 2)
        stable = (not gap and not self.reset_frame and not context.get('scene_quarantine'))
        full_box = (o['bbox'][3]-o['bbox'][1])/o['height'] >= self.new_config['min_person_height']
        eligible = scene and spatial and full_box and stable and not veto
        if eligible and kit:
            old['streak'] = old['streak']+1 if kit==old['team'] else 1
            old['team'] = kit
        else:
            old['streak'] = 0
        action = 'unchanged_baseline'
        if context.get('scene_quarantine'):
            # Preserve freshly supported referee decisions, but never display
            # a player on the first uncertain frame of a possible new scene.
            if role == 'active_player': role, team, reason = 'uncertain',None,'scene_change_quarantine'
            action = 'scene_change_quarantine'
        elif veto:
            old['confirmed']=None; action='referee_or_nonparticipant_veto'
        elif role != 'active_player' and eligible and old['streak'] >= self.new_config['uniform_confirm_frames']:
            role, team, reason = 'active_player',kit,'repeated_uniform_with_independent_floor_and_markings'
            old['confirmed']=frame; action='new_player_independent_scene'
        elif role != 'active_player':
            action = ('awaiting_repeated_uniform' if eligible else 'independent_scene_or_person_support_missing')
        old['seen']=frame;self.new_history[tid]=old
        detail.update(scene_epoch=self.scene_epoch, scene_cut_confirmed=bool(context.get('scene_cut')),
            scene_quarantine=bool(context.get('scene_quarantine')), independent_match_scene=observed_scene,
            independent_scene_recent_votes=len(self.scene_support), independent_scene_effective=scene,
            independent_person_support=spatial,independent_uniform_streak=old['streak'],
            stage1b_action=action, identity_scope=f'{self.scene_epoch}:{tid}',
            stage1b_reset_team=bool(self.reset_frame or (gap and was_new_player)), role_change_reason=reason)
        return role,team,reason,detail


class TeamsStage1B(TeamsV57):
    def __init__(self,*args):
        super().__init__(*args);self.epochs={}

    def observe_with_detail(self,tid,frame,role,vote,detail):
        epoch=detail['scene_epoch']
        if self.epochs.get(tid)!=epoch or detail['stage1b_reset_team']:
            self._clear(tid)
        self.epochs[tid]=epoch
        return super().observe_with_detail(tid,frame,role,vote,detail)
