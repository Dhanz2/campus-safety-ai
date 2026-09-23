import cv2
from collections import defaultdict
from ultralytics import YOLO
from pathlib import Path

MODEL_NAME='yolo26n.pt'
VEHICLE_CLASSES={1:'Bicycle',2:'Car',3:'Motorcycle',5:'Bus',7:'Truck'}
VEHICLE_CLASS_IDS=list(VEHICLE_CLASSES.keys())

def process_video(input_path, output_path, expected_direction='outbound', evidence_dir=None, job_id='demo'):
    model=YOLO(MODEL_NAME)
    cap=cv2.VideoCapture(input_path)
    if not cap.isOpened(): raise RuntimeError('Could not open the uploaded video.')
    fps=cap.get(cv2.CAP_PROP_FPS) or 25.0; width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280); height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    if width>1280:
        scale=1280/width; width=1280; height=int(height*scale)
    writer=cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
    if not writer.isOpened(): cap.release(); raise RuntimeError('Could not create annotated output video.')
    line_y=int(height*.60); previous={}; alerted=set(); counts=defaultdict(int); unique=set(); alerts=[]; frame_no=0
    evidence_path=Path(evidence_dir) if evidence_dir else None
    if evidence_path: evidence_path.mkdir(parents=True,exist_ok=True)
    while True:
        ok,frame=cap.read()
        if not ok: break
        if frame.shape[1]!=width or frame.shape[0]!=height: frame=cv2.resize(frame,(width,height))
        result=model.track(frame,persist=True,tracker='bytetrack.yaml',classes=VEHICLE_CLASS_IDS,conf=.35,verbose=False)[0]
        annotated=result.plot()
        cv2.line(annotated,(0,line_y),(width,line_y),(255,255,255),2)
        cv2.putText(annotated,f'MONITORED GATE | Expected: {expected_direction.upper()}',(15,max(30,line_y-12)),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
        if result.boxes is not None and len(result.boxes)>0:
            ids=result.boxes.id; xyxy=result.boxes.xyxy.cpu().tolist(); cls=result.boxes.cls.cpu().tolist(); confs=result.boxes.conf.cpu().tolist()
            track_ids=ids.int().cpu().tolist() if ids is not None else [None]*len(xyxy)
            for box,cid,conf,tid in zip(xyxy,cls,confs,track_ids):
                cid=int(cid)
                if cid not in VEHICLE_CLASSES: continue
                vehicle=VEHICLE_CLASSES[cid]; counts[vehicle]+=1
                if tid is None: continue
                unique.add(tid); x1,y1,x2,y2=map(int,box); cx=(x1+x2)//2; cy=(y1+y2)//2
                old=previous.get(tid)
                if old and tid not in alerted:
                    direction='outbound' if old[1]<line_y<=cy else ('inbound' if old[1]>line_y>=cy else None)
                    if direction and direction!=expected_direction:
                        evidence_url=None
                        if evidence_path:
                            name=f'{job_id}_incident_{tid}_{frame_no}.jpg'; path=evidence_path/name; cv2.imwrite(str(path),annotated); evidence_url='/static/evidence/'+name
                        alerts.append({'type':'Traffic','description':f'Possible wrong-direction movement detected: {vehicle} track #{tid} ({direction}; expected {expected_direction}).','track_id':int(tid),'vehicle':vehicle,'direction':direction,'confidence':round(float(conf),2),'frame':frame_no,'evidence_url':evidence_url})
                        alerted.add(tid)
                previous[tid]=(cx,cy)
        y=30
        for name in ('Car','Motorcycle','Bus','Truck','Bicycle'):
            cv2.putText(annotated,f'{name}: {counts[name]}',(15,y),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2); y+=22
        cv2.putText(annotated,f'Frame {frame_no}',(15,height-18),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2)
        frame_no+=1; writer.write(annotated)
    cap.release(); writer.release()
    return {'frames':frame_no,'unique_vehicles':len(unique),'vehicle_counts':dict(counts),'alerts':alerts,'output_path':output_path}
