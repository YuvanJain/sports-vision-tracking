"""Scene-scoped court duty and structured NEW-referee confirmation.

No identity-specific exceptions. Existing confirmed-referee persistence and
Stage 1B player confirmation are inherited unchanged.
"""
from collections import deque
from .basketball_stage1b import PlayerStage1B, TeamsStage1B, FIELDS_1B
from .basketball_v57 import FIELDS_V57
from .basketball_v52 import snapshot_from_evidence

FIELDS_1C = ('stage1c_boundary_invalidated', 'stage1c_current_scene_court',
    'stage1c_new_referee_structure', 'stage1c_referee_structure_blocked',
    'stage1c_raw_stripe_transitions', 'stage1c_effective_court_reason')


class PlayerStage1C(PlayerStage1B):
    def __init__(self, config):
        super().__init__(config)
        self.current_scene_court = set()

    def observe(self, observation, context):
        tid=observation['track_id']
        o=dict(observation)
        a=o['appearance']
        # In particular, historical replay may still supply boundary grace
        # inherited from a prior view. Live CourtEvidence also resets on cuts.
        if (o['court_reason']=='court' and not context.get('scene_quarantine') and
                float(o.get('court_fraction',a.get('court_fraction',0)))>0):
            self.current_scene_court.add(tid)
        invalid=(self.scene_epoch>0 and o['court_reason']=='boundary_grace' and
                 tid not in self.current_scene_court)
        if invalid:
            o['court_reason']='mask_uncertain' if o.get('court_mask_status')=='uncertain' else 'outside_court'
            o['interior']=False
            if o['court_reason']=='mask_uncertain':o['playline']=None
            o['instant'],o['raw_team'],_=snapshot_from_evidence(a,o['court_reason'],o['foot'],o['playline'],self.full_config['participant'])
        confirmed=(self.ref_state.get(tid,{}).get('confirmed',False) or
                   self.closeup_state.get(tid,{}).get('confirmed',False))
        minimum=self.full_config['closeup_referee_v56_final']['min_center_stripe_transitions']
        structure=(a.get('center_stripe_transitions',0)>=minimum or
            (a.get('stripe_evidence',False) and a.get('stripe_transitions',0)>=minimum and
             a.get('stripe_dark_runs',0)>=2))
        role,team,reason,detail=super().observe(o,context)
        # This is a close-up/post-cut correction, not a global increase in
        # referee thresholds. Reliable current-scene court evidence continues
        # to use the original partially-occluded referee confirmation rule.
        unsupported_duty=(tid not in self.current_scene_court and
                          (self.scene_epoch>0 or context.get('scene_visual_available',False)))
        blocked=bool(role=='referee' and not confirmed and not structure and unsupported_duty)
        if blocked:
            # One dark object against a white shirt is not a repeated stripe
            # pattern. Do not let these rejected votes seed later confirmation.
            role,team,reason='uncertain',None,'new_referee_requires_repeated_stripe_structure'
            self.ref_state.pop(tid,None)
            self.appearance_history.pop(tid,None)
            self.closeup_state.pop(tid,None)
            self.closeup_history.pop(tid,None)
            if tid in self.history:
                history=self.history[tid]
                self.history[tid]=deque(((f,p,False,t,fg,s,m) for f,p,r,t,fg,s,m in history),maxlen=history.maxlen)
            detail.update(ref_confirmed=False,ref_state_action='unstructured_new_referee_rejected',
                          closeup_confirmed=False,role_change_reason=reason)
        detail.update(stage1c_boundary_invalidated=invalid,
            stage1c_current_scene_court=tid in self.current_scene_court,
            stage1c_new_referee_structure=bool(structure),
            stage1c_referee_structure_blocked=blocked,
            stage1c_raw_stripe_transitions=a.get('center_stripe_transitions',0),
            stage1c_effective_court_reason=o['court_reason'])
        return role,team,reason,detail


TeamsStage1C = TeamsStage1B
