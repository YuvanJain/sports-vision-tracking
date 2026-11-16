"""Scene-local close-up player support from a repeated opposing-uniform group.

No identity labels enter inference. Group geometry supplies context independent
of the candidate's own jersey; it is a heuristic, not proof of active duty.
Existing referee/nonparticipant decisions take precedence, unchanged.
"""
from collections import deque
from .basketball_stage1c import PlayerStage1C, TeamsStage1C

FIELDS_1E=('stage1e_action','stage1e_uniform_team','stage1e_uniform_streak',
    'stage1e_context_peer_ids','stage1e_context_votes','stage1e_context_age',
    'stage1e_context_current','stage1e_veto','stage1e_identity_reset')

class PlayerStage1E(PlayerStage1C):
    def __init__(self,config):
        super().__init__(config)
        self.e=config['player_stage1e']
        self.group_memory={}
        self.peer_ids={}

    def peer_uniform(self,o):
        """A context proposal only; never assign a role/team from this helper."""
        a=o['appearance'];u=self.uniform_config
        if not a.get('crop_valid') or not a.get('lower_valid',False):return None
        # Retain the existing upper/lower uniform definitions. A strong target
        # vote is still mandatory at promotion; context peers cannot be promoted
        # by these weaker proposals or by the group's own output predictions.
        b=a.get('purple_fraction',0)>=u['b_shirt_min'] and max(a.get('lower_purple',0),a.get('lower_yellow',0))>=u['b_lower_min']
        white=a.get('white_fraction',0)>=u['a_shirt_min'] and a.get('lower_white',0)>=u['a_lower_min']
        if a.get('center_stripe_transitions',0)>=2 and a.get('center_stripe_dark',0)>=.15:return None
        return 'B' if b and not white else 'A' if white and not b else None

    def standing_geometry(self,o):
        x1,y1,x2,y2=o['bbox'];w,h=o['width'],o['height'];bh=y2-y1
        return (bh/h>=self.e['min_height_fraction'] and
            self.e['min_aspect']<=(x2-x1)/max(1,bh)<=self.e['max_aspect'] and
            x1/w>self.e['edge_margin'] and x2/w<1-self.e['edge_margin'] and
            y1/h>self.e['edge_margin'])

    @staticmethod
    def overlap(a,b):
        x=max(0,min(a[2],b[2])-max(a[0],b[0]));y=max(0,min(a[3],b[3])-max(a[1],b[1]))
        inter=x*y
        return inter/max(1,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)

    def begin_frame(self,observations,context):
        super().begin_frame(observations,context)
        self.peer_ids={}
        if self.reset_frame or not context.get('scene_visual_available'):return
        proposals=[(o,self.peer_uniform(o)) for o in observations if self.standing_geometry(o)]
        proposals=[(o,t) for o,t in proposals if t]
        for o,team in proposals:
            peers=[]
            for other,ot in proposals:
                if o['track_id']==other['track_id'] or team==ot:continue
                heights=[(q['bbox'][3]-q['bbox'][1])/q['height'] for q in (o,other)]
                if (max(heights)/min(heights)<=self.e['max_peer_height_ratio'] and
                    abs(o['foot']-other['foot'])<=self.e['max_peer_foot_difference'] and
                    self.overlap(o['bbox'],other['bbox'])<=self.e['max_peer_overlap']):
                    peers.append(int(other['track_id']))
            self.peer_ids[int(o['track_id'])]=peers

    def observe(self,o,context):
        role,team,reason,detail=super().observe(o,context)
        tid,frame=int(o['track_id']),int(o['frame_id']);geom=self.geometry(o)
        old=self.group_memory.get(tid)
        reset=(old is None or self.reset_frame or frame-old['seen']>self.e['max_observation_gap'] or
            self.discontinuous(old['geometry'],geom))
        if reset:old=dict(seen=frame,geometry=geom,team=None,streak=0,support=deque())
        kit=self.uniform(o['appearance'])
        peers=self.peer_ids.get(tid,[])
        # No change to hard foreground memory, referee precedence or table rules.
        veto=(role in ('referee','table_official','nonparticipant') or detail.get('referee_appearance') or
              detail.get('foreground_candidate') or detail.get('foreground_memory_active') or
              detail.get('table_position') or context.get('table_group_count',0)>=2)
        eligible=(not veto and not self.reset_frame and not context.get('scene_quarantine') and
            context.get('scene_visual_available') and self.standing_geometry(o) and
            o['foot']<self.e['max_target_foot'] and o['court_reason']=='mask_uncertain')
        while old['support'] and frame-old['support'][0]>self.e['context_window_frames']:old['support'].popleft()
        if eligible and kit:
            old['streak']=old['streak']+1 if kit==old['team'] else 1
            if old['team'] not in (None,kit):old['support'].clear()
            old['team']=kit
            if peers:old['support'].append(frame)
        else:
            old['streak']=0
            if veto or self.reset_frame:old['support'].clear()
        age=frame-old['support'][-1] if old['support'] else None
        supported=(len(old['support'])>=self.e['min_context_observations'] and age is not None and age<=self.e['context_hold_frames'])
        action='baseline_unchanged'
        if role=='uncertain' and eligible and kit:
            if supported and old['streak']>=self.e['uniform_confirm_frames']:
                role,team,reason='active_player',kit,'repeated_full_uniform_and_opposing_standing_group'
                action='closeup_group_confirmed'
            else:action='awaiting_uniform_and_independent_group'
        elif veto:action='existing_referee_or_nonparticipant_veto'
        old.update(seen=frame,geometry=geom);self.group_memory[tid]=old
        detail.update(stage1e_action=action,stage1e_uniform_team=kit,stage1e_uniform_streak=old['streak'],
            stage1e_context_peer_ids=peers,stage1e_context_votes=len(old['support']),stage1e_context_age=age,
            stage1e_context_current=bool(peers),stage1e_veto=bool(veto),stage1e_identity_reset=reset,
            role_change_reason=reason)
        return role,team,reason,detail

TeamsStage1E=TeamsStage1C
