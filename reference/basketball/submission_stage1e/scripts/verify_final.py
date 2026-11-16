"""Read only check of completed five-chunk exports; never invokes inference."""
import csv
import json
from pathlib import Path
import cv2

REQUIRED=('annotated.mp4','tracks.csv','tracks.json','detections.jsonl','metrics.json')

def verify(root):
    root=Path(root)
    all_ids=set();chunks=[];previous_max=-1
    for i in range(5):
        folder=root/f'chunk_{i+1:02d}';start=900+i*3600;end=start+3600
        for name in REQUIRED:
            p=folder/name
            if not p.is_file() or p.stat().st_size==0:raise RuntimeError(f'Missing/empty: {p}')
        video=cv2.VideoCapture(str(folder/'annotated.mp4'))
        try:
            if not video.isOpened() or int(video.get(cv2.CAP_PROP_FRAME_COUNT))!=3600:
                raise RuntimeError(f'{folder}: video unreadable or frame count differs')
        finally:video.release()
        metric=json.loads((folder/'metrics.json').read_text())
        if (metric['start_frame'],metric['end_frame_exclusive'],metric['frames_processed'])!=(start,end,3600):
            raise RuntimeError(f'Chunk {i+1} metrics do not match frame interval')
        frames=[];ids=set();rows=0
        with (folder/'tracks.csv').open(newline='',encoding='utf-8') as file:
            reader=csv.DictReader(file)
            for key in ('frame_id','timestamp','track_id','class_name','bbox','team_id','event','shot_outcome'):
                if key not in reader.fieldnames:raise RuntimeError(f'{folder}: missing {key}')
            for row in reader:
                frame=int(row['frame_id']);frames.append(frame);rows+=1
                if row['track_id'] and row['class_name']!='ball':ids.add(int(row['track_id']))
                if row['event'] or row['shot_outcome']:raise RuntimeError('Unverified shot label in export')
                if abs(float(row['timestamp'])-frame/30)>1e-3:raise RuntimeError('Timestamp mismatch')
        if sorted(set(frames))!=list(range(start,end)):raise RuntimeError(f'{folder}: CSV frame gap/overlap')
        with (folder/'detections.jsonl').open(encoding='utf-8') as file:
            diagnostic_frames=[json.loads(line)['frame_id'] for line in file]
        if diagnostic_frames!=list(range(start,end)):raise RuntimeError(f'{folder}: diagnostics frame gap/overlap')
        # A monotone maximum is evidence against numbering reset, not proof of identity accuracy.
        if ids and previous_max>=0 and max(ids)<previous_max:raise RuntimeError('IDs may have reset')
        if ids:previous_max=max(ids)
        all_ids.update(ids)
        chunks.append({'chunk':i+1,'start':start,'end_exclusive':end,'csv_rows':rows,'unique_person_ids':len(ids),
                       'max_person_id':max(ids) if ids else None})
    overall=json.loads((root/'metrics.json').read_text())
    if overall['frames_processed']!=18000 or overall['unique_person_track_ids']!=len(all_ids):
        raise RuntimeError('Overall metrics do not reconcile with chunk IDs')
    return {'status':'STRUCTURAL PASS; visual tracking accuracy not established','frames':18000,
            'unique_person_ids':len(all_ids),'chunks':chunks}
