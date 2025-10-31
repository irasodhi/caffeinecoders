from flask import Flask, render_template, Response
import cv2
import numpy as np
import mediapipe as mp
import time
import winsound  # only works on Windows

app = Flask(__name__)

# ---------- SETTINGS ----------
MAR_YAWN_THRESHOLD = 0.6
EAR_BLINK_THRESHOLD = 0.23
EAR_SLEEP_THRESHOLD = 0.23
SLEEP_SECONDS = 2.0
ALERT_FREQ = 1200
ALERT_DUR = 800
# ------------------------------

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def mouth_aspect_ratio(landmarks, w, h):
    up = (landmarks[13].x * w, landmarks[13].y * h)
    low = (landmarks[14].x * w, landmarks[14].y * h)
    left = (landmarks[61].x * w, landmarks[61].y * h)
    right = (landmarks[291].x * w, landmarks[291].y * h)
    return euclidean(up, low) / euclidean(left, right)

def eye_aspect_ratio(landmarks, w, h, side="left"):
    if side == "right":
        ids = [33, 160, 158, 133, 153, 144]
    else:
        ids = [362, 385, 387, 263, 380, 373]
    p = [(landmarks[i].x * w, landmarks[i].y * h) for i in ids]
    vertical = (euclidean(p[1], p[5]) + euclidean(p[2], p[4])) / 2
    horizontal = euclidean(p[0], p[3])
    return vertical / horizontal

def generate_frames():
    cap = cv2.VideoCapture(0)
    yawns = 0
    blinks = 0
    blink_active = False
    yawn_active = False
    sleep_start_time = None
    sleeping = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]
            landmarks = face.landmark

            left_ear = eye_aspect_ratio(landmarks, w, h, "left")
            right_ear = eye_aspect_ratio(landmarks, w, h, "right")
            ear = (left_ear + right_ear) / 2
            mar = mouth_aspect_ratio(landmarks, w, h)

            # blink logic
            if ear < EAR_BLINK_THRESHOLD:
                if not blink_active:
                    blink_active = True
            else:
                if blink_active:
                    blinks += 1
                    blink_active = False

            # yawn logic
            if mar > MAR_YAWN_THRESHOLD:
                if not yawn_active:
                    yawns += 1
                    yawn_active = True
            else:
                yawn_active = False

            # sleep logic
            if ear < EAR_SLEEP_THRESHOLD:
                if sleep_start_time is None:
                    sleep_start_time = time.time()
                elif (time.time() - sleep_start_time) >= SLEEP_SECONDS:
                    sleeping = True
            else:
                sleep_start_time = None
                sleeping = False

            # draw text on frame
            if sleeping:
                cv2.putText(frame, "⚠️ DROWSINESS ALERT!", (60, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
                winsound.Beep(ALERT_FREQ, ALERT_DUR)
            else:
                cv2.putText(frame, "Status: Awake", (60, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)

            cv2.putText(frame, f"EAR: {ear:.2f}  MAR: {mar:.2f}", (20, h - 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Blinks: {blinks}  Yawns: {yawns}", (20, h - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # encode to JPEG and yield frame for the web stream
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



from flask import Flask, render_template, request, jsonify
import json



# F1 Cars Database (2010-2024)
F1_CARS = {
    'Red Bull RB19': {
        'team': 'Red Bull Racing', 'year': 2023, 'downforce': 95, 'topSpeed': 350,
        'acceleration': 94, 'cornering': 98, 'reliability': 96, 'wetPerformance': 92,
        'straightSpeed': 88, 'aero': 98, 'powerUnit': 'Honda RBPT', 'championships': 1, 'image': '🏎'
    },
    'Red Bull RB18': {
        'team': 'Red Bull Racing', 'year': 2022, 'downforce': 93, 'topSpeed': 345,
        'acceleration': 92, 'cornering': 96, 'reliability': 94, 'wetPerformance': 90,
        'straightSpeed': 86, 'aero': 96, 'powerUnit': 'Honda RBPT', 'championships': 1, 'image': '🏎'
    },
    'Mercedes W13': {
        'team': 'Mercedes AMG', 'year': 2022, 'downforce': 88, 'topSpeed': 340,
        'acceleration': 89, 'cornering': 87, 'reliability': 91, 'wetPerformance': 93,
        'straightSpeed': 92, 'aero': 85, 'powerUnit': 'Mercedes', 'championships': 0, 'image': '🏎'
    },
    'Mercedes W11': {
        'team': 'Mercedes AMG', 'year': 2020, 'downforce': 96, 'topSpeed': 348,
        'acceleration': 95, 'cornering': 97, 'reliability': 98, 'wetPerformance': 96,
        'straightSpeed': 89, 'aero': 97, 'powerUnit': 'Mercedes', 'championships': 1, 'image': '🏎'
    },
    'Ferrari F1-75': {
        'team': 'Scuderia Ferrari', 'year': 2022, 'downforce': 91, 'topSpeed': 352,
        'acceleration': 93, 'cornering': 90, 'reliability': 85, 'wetPerformance': 88,
        'straightSpeed': 95, 'aero': 92, 'powerUnit': 'Ferrari', 'championships': 0, 'image': '🏎'
    },
    'Ferrari SF90': {
        'team': 'Scuderia Ferrari', 'year': 2019, 'downforce': 89, 'topSpeed': 355,
        'acceleration': 91, 'cornering': 88, 'reliability': 82, 'wetPerformance': 85,
        'straightSpeed': 97, 'aero': 88, 'powerUnit': 'Ferrari', 'championships': 0, 'image': '🏎'
    },
    'McLaren MCL60': {
        'team': 'McLaren F1', 'year': 2023, 'downforce': 89, 'topSpeed': 342,
        'acceleration': 90, 'cornering': 91, 'reliability': 92, 'wetPerformance': 89,
        'straightSpeed': 87, 'aero': 90, 'powerUnit': 'Mercedes', 'championships': 0, 'image': '🏎'
    },
    'McLaren MP4-25': {
        'team': 'McLaren F1', 'year': 2010, 'downforce': 85, 'topSpeed': 320,
        'acceleration': 84, 'cornering': 87, 'reliability': 88, 'wetPerformance': 86,
        'straightSpeed': 82, 'aero': 86, 'powerUnit': 'Mercedes', 'championships': 0, 'image': '🏎'
    },
    'Red Bull RB6': {
        'team': 'Red Bull Racing', 'year': 2010, 'downforce': 92, 'topSpeed': 318,
        'acceleration': 87, 'cornering': 94, 'reliability': 90, 'wetPerformance': 88,
        'straightSpeed': 80, 'aero': 94, 'powerUnit': 'Renault', 'championships': 1, 'image': '🏎'
    },
    'Red Bull RB7': {
        'team': 'Red Bull Racing', 'year': 2011, 'downforce': 94, 'topSpeed': 322,
        'acceleration': 89, 'cornering': 96, 'reliability': 91, 'wetPerformance': 90,
        'straightSpeed': 81, 'aero': 96, 'powerUnit': 'Renault', 'championships': 1, 'image': '🏎'
    },
    'Mercedes W05': {
        'team': 'Mercedes AMG', 'year': 2014, 'downforce': 90, 'topSpeed': 335,
        'acceleration': 92, 'cornering': 93, 'reliability': 95, 'wetPerformance': 94,
        'straightSpeed': 88, 'aero': 92, 'powerUnit': 'Mercedes', 'championships': 1, 'image': '🏎'
    },
    'Red Bull RB16B': {
        'team': 'Red Bull Racing', 'year': 2021, 'downforce': 92, 'topSpeed': 343,
        'acceleration': 91, 'cornering': 95, 'reliability': 93, 'wetPerformance': 91,
        'straightSpeed': 85, 'aero': 94, 'powerUnit': 'Honda', 'championships': 1, 'image': '🏎'
    },
    'Mercedes W12': {
        'team': 'Mercedes AMG', 'year': 2021, 'downforce': 91, 'topSpeed': 341,
        'acceleration': 90, 'cornering': 94, 'reliability': 95, 'wetPerformance': 95,
        'straightSpeed': 87, 'aero': 93, 'powerUnit': 'Mercedes', 'championships': 0, 'image': '🏎'
    },
    'Ferrari SF71H': {
        'team': 'Scuderia Ferrari', 'year': 2018, 'downforce': 90, 'topSpeed': 349,
        'acceleration': 92, 'cornering': 91, 'reliability': 86, 'wetPerformance': 88,
        'straightSpeed': 94, 'aero': 91, 'powerUnit': 'Ferrari', 'championships': 0, 'image': '🏎'
    },
    'Mercedes W07': {
        'team': 'Mercedes AMG', 'year': 2016, 'downforce': 92, 'topSpeed': 340,
        'acceleration': 93, 'cornering': 95, 'reliability': 96, 'wetPerformance': 95,
        'straightSpeed': 87, 'aero': 94, 'powerUnit': 'Mercedes', 'championships': 1, 'image': '🏎'
    },
    'Alpine A522': {
        'team': 'Alpine F1', 'year': 2022, 'downforce': 86, 'topSpeed': 338,
        'acceleration': 87, 'cornering': 88, 'reliability': 89, 'wetPerformance': 87,
        'straightSpeed': 86, 'aero': 87, 'powerUnit': 'Renault', 'championships': 0, 'image': '🏎'
    },
    'Aston Martin AMR23': {
        'team': 'Aston Martin', 'year': 2023, 'downforce': 90, 'topSpeed': 344,
        'acceleration': 91, 'cornering': 92, 'reliability': 90, 'wetPerformance': 89,
        'straightSpeed': 88, 'aero': 91, 'powerUnit': 'Mercedes', 'championships': 0, 'image': '🏎'
    },
    'Ferrari F2004': {
        'team': 'Scuderia Ferrari', 'year': 2004, 'downforce': 88, 'topSpeed': 365,
        'acceleration': 90, 'cornering': 91, 'reliability': 97, 'wetPerformance': 87,
        'straightSpeed': 98, 'aero': 89, 'powerUnit': 'Ferrari', 'championships': 1, 'image': '🏎'
    }
}

def calculate_score(car, conditions):
    """Calculate performance score based on race conditions"""
    score = 0
    factors = []
    
    # Weather impact
    if conditions['weather'] in ['wet', 'rain']:
        score += car['wetPerformance'] * 1.5
        factors.append({'name': 'Wet Weather', 'value': car['wetPerformance'], 'weight': 1.5})
    else:
        score += car['topSpeed'] * 0.8
        factors.append({'name': 'Dry Performance', 'value': car['topSpeed'] / 4, 'weight': 0.8})
    
    # Track type impact
    if conditions['trackType'] == 'street':
        score += car['cornering'] * 1.3
        score += car['acceleration'] * 1.2
        factors.append({'name': 'Cornering', 'value': car['cornering'], 'weight': 1.3})
    elif conditions['trackType'] == 'high-speed':
        score += car['straightSpeed'] * 1.5
        score += car['topSpeed'] * 0.01
        factors.append({'name': 'Top Speed', 'value': car['topSpeed'] / 4, 'weight': 1.5})
    else:
        score += car['downforce'] * 1.2
        factors.append({'name': 'Downforce', 'value': car['downforce'], 'weight': 1.2})
    
    # Waviness impact
    if conditions['waviness'] == 'bumpy':
        score += car['reliability'] * 1.1
        factors.append({'name': 'Reliability', 'value': car['reliability'], 'weight': 1.1})
    
    # Curves impact
    if conditions['curves'] == 'high':
        score += car['cornering'] * 1.4
        score += car['aero'] * 1.2
    elif conditions['curves'] == 'low':
        score += car['straightSpeed'] * 1.3
    
    # Race duration impact
    duration = int(conditions['raceDuration'])
    if duration >= 12:
        score += car['reliability'] * 1.5
        factors.append({'name': 'Endurance', 'value': car['reliability'], 'weight': 1.5})
    else:
        score += car['acceleration'] * 1.2
    
    return {'score': score, 'factors': factors}

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze race conditions and return recommendations"""
    conditions = request.json
    
    # Calculate scores for all cars
    results = []
    for name, car in F1_CARS.items():
        calc_result = calculate_score(car, conditions)
        results.append({
            'name': name,
            'car': car,
            'score': calc_result['score'],
            'factors': calc_result['factors']
        })
    
    # Sort by score
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # Return top 5 and all results
    return jsonify({
        'recommended': results[0],
        'alternatives': results[1:5],
        'allCars': results
    })
@app.route('/focus_detector')
def focus_detector():
    return render_template('home1.html')

@app.route('/index')
def index():
    return render_template('index.html')  
if __name__ == '_main_':
    app.run(debug=True, host='0.0.0.0', port=5000)