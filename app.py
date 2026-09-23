from flask import Flask, render_template, jsonify, request
from datetime import datetime
from pathlib import Path
from werkzeug.utils import secure_filename
import sqlite3
import uuid

from ai.video_detector import process_video

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024
BASE_DIR = Path(__file__).resolve().parent
DB = BASE_DIR / "campus.db"
UPLOAD_DIR = BASE_DIR / "uploads"
PROCESSED_DIR = BASE_DIR / "static" / "processed"
EVIDENCE_DIR = BASE_DIR / "static" / "evidence"
for p in (UPLOAD_DIR, PROCESSED_DIR, EVIDENCE_DIR): p.mkdir(parents=True, exist_ok=True)
ALLOWED_VIDEO = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

def init_db():
    con=sqlite3.connect(DB); cur=con.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS incidents (
      id INTEGER PRIMARY KEY AUTOINCREMENT, time TEXT NOT NULL, location TEXT NOT NULL,
      category TEXT NOT NULL, description TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'Pending Review',
      vehicle_type TEXT, track_id INTEGER, direction TEXT, evidence_url TEXT, video_url TEXT)''')
    cols={r[1] for r in cur.execute('PRAGMA table_info(incidents)').fetchall()}
    for name,typ in [("vehicle_type","TEXT"),("track_id","INTEGER"),("direction","TEXT"),("evidence_url","TEXT"),("video_url","TEXT")]:
        if name not in cols: cur.execute(f'ALTER TABLE incidents ADD COLUMN {name} {typ}')
    con.commit(); con.close()

def seed_demo():
    con=sqlite3.connect(DB); cur=con.cursor()
    if cur.execute('SELECT COUNT(*) FROM incidents').fetchone()[0]==0:
        now=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cur.execute('INSERT INTO incidents(time,location,category,description,status) VALUES(?,?,?,?,?)',
                    (now,'Main Gate','Traffic','Demo traffic event awaiting human verification.','Pending Review'))
        cur.execute('INSERT INTO incidents(time,location,category,description,status) VALUES(?,?,?,?,?)',
                    (now,'Canteen Road','Safety','Potential altercation detected. Human verification required.','Pending Review'))
        con.commit()
    con.close()

def add_incident(location, category, description, **extra):
    con=sqlite3.connect(DB); cur=con.cursor()
    cur.execute('''INSERT INTO incidents(time,location,category,description,status,vehicle_type,track_id,direction,evidence_url,video_url)
                   VALUES(?,?,?,?,?,?,?,?,?,?)''',
                (datetime.now().strftime('%Y-%m-%d %H:%M:%S'),location,category,description,'Pending Review',
                 extra.get('vehicle_type'),extra.get('track_id'),extra.get('direction'),extra.get('evidence_url'),extra.get('video_url')))
    con.commit(); iid=cur.lastrowid; con.close(); return iid

@app.route('/')
def index(): return render_template('index.html')

@app.get('/api/incidents')
def incidents():
    con=sqlite3.connect(DB); con.row_factory=sqlite3.Row
    rows=con.execute('SELECT * FROM incidents ORDER BY id DESC').fetchall(); con.close()
    return jsonify([dict(r) for r in rows])

@app.get('/api/stats')
def stats():
    con=sqlite3.connect(DB)
    d={'total':con.execute('SELECT COUNT(*) FROM incidents').fetchone()[0],
       'pending':con.execute("SELECT COUNT(*) FROM incidents WHERE status='Pending Review'").fetchone()[0],
       'traffic':con.execute("SELECT COUNT(*) FROM incidents WHERE category='Traffic'").fetchone()[0],
       'safety':con.execute("SELECT COUNT(*) FROM incidents WHERE category='Safety'").fetchone()[0]}
    con.close(); return jsonify(d)

@app.patch('/api/incidents/<int:incident_id>')
def update_incident(incident_id):
    status=request.get_json(force=True).get('status'); allowed={'Pending Review','Verified','Dismissed','Resolved'}
    if status not in allowed: return jsonify({'error':'Invalid status'}),400
    con=sqlite3.connect(DB); cur=con.cursor(); cur.execute('UPDATE incidents SET status=? WHERE id=?',(status,incident_id)); con.commit(); changed=cur.rowcount; con.close()
    return (jsonify({'message':'Updated'}) if changed else (jsonify({'error':'Incident not found'}),404))

@app.post('/api/analyze-video')
def analyze_video():
    video=request.files.get('video'); location=request.form.get('location','Main Gate').strip() or 'Main Gate'
    expected=request.form.get('expected_direction','outbound')
    if expected not in {'inbound','outbound'}: expected='outbound'
    if not video or not video.filename: return jsonify({'error':'Please choose a video file.'}),400
    suffix=Path(video.filename).suffix.lower()
    if suffix not in ALLOWED_VIDEO: return jsonify({'error':'Unsupported video type. Use MP4, AVI, MOV, MKV or WEBM.'}),400
    job=uuid.uuid4().hex[:10]; input_path=UPLOAD_DIR/f'{job}_{secure_filename(video.filename)}'
    output_path=PROCESSED_DIR/f'{job}_annotated.mp4'; video.save(input_path)
    try:
        result=process_video(str(input_path),str(output_path),expected_direction=expected,evidence_dir=str(EVIDENCE_DIR),job_id=job)
        video_url=f'/static/processed/{output_path.name}'
        for a in result['alerts']:
            add_incident(location,'Traffic',a['description']+' Human verification required.',
                         vehicle_type=a['vehicle'],track_id=a['track_id'],direction=a['direction'],
                         evidence_url=a['evidence_url'],video_url=video_url)
        return jsonify({'message':'Video analysis complete','video_url':video_url,'video_download':video_url,
                        'frames':result['frames'],'unique_vehicles':result['unique_vehicles'],
                        'vehicle_counts':result['vehicle_counts'],'alerts':result['alerts']})
    except Exception as exc:
        return jsonify({'error':f'Video analysis failed: {exc}'}),500
    finally:
        input_path.unlink(missing_ok=True)

if __name__=='__main__':
    init_db(); seed_demo(); app.run(debug=True)
