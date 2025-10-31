# from flask import Flask, render_template, Response
# import cv2
# import numpy as np
# import mediapipe as mp
# import time
# import winsound  # only works on Windows

# app = Flask(__name__)

# # ---------- SETTINGS ----------
# MAR_YAWN_THRESHOLD = 0.6
# EAR_BLINK_THRESHOLD = 0.23
# EAR_SLEEP_THRESHOLD = 0.23
# SLEEP_SECONDS = 2.0
# ALERT_FREQ = 1200
# ALERT_DUR = 800
# # ------------------------------

# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(
#     max_num_faces=1,
#     refine_landmarks=True,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )

# def euclidean(p1, p2):
#     return np.linalg.norm(np.array(p1) - np.array(p2))

# def mouth_aspect_ratio(landmarks, w, h):
#     up = (landmarks[13].x * w, landmarks[13].y * h)
#     low = (landmarks[14].x * w, landmarks[14].y * h)
#     left = (landmarks[61].x * w, landmarks[61].y * h)
#     right = (landmarks[291].x * w, landmarks[291].y * h)
#     return euclidean(up, low) / euclidean(left, right)

# def eye_aspect_ratio(landmarks, w, h, side="left"):
#     if side == "right":
#         ids = [33, 160, 158, 133, 153, 144]
#     else:
#         ids = [362, 385, 387, 263, 380, 373]
#     p = [(landmarks[i].x * w, landmarks[i].y * h) for i in ids]
#     vertical = (euclidean(p[1], p[5]) + euclidean(p[2], p[4])) / 2
#     horizontal = euclidean(p[0], p[3])
#     return vertical / horizontal

# def generate_frames():
#     cap = cv2.VideoCapture(0)
#     yawns = 0
#     blinks = 0
#     blink_active = False
#     yawn_active = False
#     sleep_start_time = None
#     sleeping = False

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         h, w = frame.shape[:2]
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = face_mesh.process(rgb)

#         if results.multi_face_landmarks:
#             face = results.multi_face_landmarks[0]
#             landmarks = face.landmark

#             left_ear = eye_aspect_ratio(landmarks, w, h, "left")
#             right_ear = eye_aspect_ratio(landmarks, w, h, "right")
#             ear = (left_ear + right_ear) / 2
#             mar = mouth_aspect_ratio(landmarks, w, h)

#             # blink logic
#             if ear < EAR_BLINK_THRESHOLD:
#                 if not blink_active:
#                     blink_active = True
#             else:
#                 if blink_active:
#                     blinks += 1
#                     blink_active = False

#             # yawn logic
#             if mar > MAR_YAWN_THRESHOLD:
#                 if not yawn_active:
#                     yawns += 1
#                     yawn_active = True
#             else:
#                 yawn_active = False

#             # sleep logic
#             if ear < EAR_SLEEP_THRESHOLD:
#                 if sleep_start_time is None:
#                     sleep_start_time = time.time()
#                 elif (time.time() - sleep_start_time) >= SLEEP_SECONDS:
#                     sleeping = True
#             else:
#                 sleep_start_time = None
#                 sleeping = False

#             # draw text on frame
#             if sleeping:
#                 cv2.putText(frame, "⚠️ DROWSINESS ALERT!", (60, 60),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
#                 winsound.Beep(ALERT_FREQ, ALERT_DUR)
#             else:
#                 cv2.putText(frame, "Status: Awake", (60, 60),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)

#             cv2.putText(frame, f"EAR: {ear:.2f}  MAR: {mar:.2f}", (20, h - 80),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
#             cv2.putText(frame, f"Blinks: {blinks}  Yawns: {yawns}", (20, h - 50),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

#         # encode to JPEG and yield frame for the web stream
#         _, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()

#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

#     cap.release()

# @app.route('/')
# def home():
#     return render_template('home.html')

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')



# from flask import Flask, render_template, request, jsonify
# import json



# # F1 Cars Database (2010-2024)
# F1_CARS = {
#     'Red Bull RB19': {
#         'team': 'Red Bull Racing', 'year': 2023, 'downforce': 95, 'topSpeed': 350,
#         'acceleration': 94, 'cornering': 98, 'reliability': 96, 'wetPerformance': 92,
#         'straightSpeed': 88, 'aero': 98, 'powerUnit': 'Honda RBPT', 'championships': 1, 'image': '🏎'
#     },
#     'Red Bull RB18': {
#         'team': 'Red Bull Racing', 'year': 2022, 'downforce': 93, 'topSpeed': 345,
#         'acceleration': 92, 'cornering': 96, 'reliability': 94, 'wetPerformance': 90,
#         'straightSpeed': 86, 'aero': 96, 'powerUnit': 'Honda RBPT', 'championships': 1, 'image': '🏎'
#     },
#     'Mercedes W13': {
#         'team': 'Mercedes AMG', 'year': 2022, 'downforce': 88, 'topSpeed': 340,
#         'acceleration': 89, 'cornering': 87, 'reliability': 91, 'wetPerformance': 93,
#         'straightSpeed': 92, 'aero': 85, 'powerUnit': 'Mercedes', 'championships': 0, 'image': '🏎'
#     },
#     'Mercedes W11': {
#         'team': 'Mercedes AMG', 'year': 2020, 'downforce': 96, 'topSpeed': 348,
#         'acceleration': 95, 'cornering': 97, 'reliability': 98, 'wetPerformance': 96,
#         'straightSpeed': 89, 'aero': 97, 'powerUnit': 'Mercedes', 'championships': 1, 'image': '🏎'
#     },
#     'Ferrari F1-75': {
#         'team': 'Scuderia Ferrari', 'year': 2022, 'downforce': 91, 'topSpeed': 352,
#         'acceleration': 93, 'cornering': 90, 'reliability': 85, 'wetPerformance': 88,
#         'straightSpeed': 95, 'aero': 92, 'powerUnit': 'Ferrari', 'championships': 0, 'image': '🏎'
#     },
#     'Ferrari SF90': {
#         'team': 'Scuderia Ferrari', 'year': 2019, 'downforce': 89, 'topSpeed': 355,
#         'acceleration': 91, 'cornering': 88, 'reliability': 82, 'wetPerformance': 85,
#         'straightSpeed': 97, 'aero': 88, 'powerUnit': 'Ferrari', 'championships': 0, 'image': '🏎'
#     },
#     'McLaren MCL60': {
#         'team': 'McLaren F1', 'year': 2023, 'downforce': 89, 'topSpeed': 342,
#         'acceleration': 90, 'cornering': 91, 'reliability': 92, 'wetPerformance': 89,
#         'straightSpeed': 87, 'aero': 90, 'powerUnit': 'Mercedes', 'championships': 0, 'image': '🏎'
#     },
#     'McLaren MP4-25': {
#         'team': 'McLaren F1', 'year': 2010, 'downforce': 85, 'topSpeed': 320,
#         'acceleration': 84, 'cornering': 87, 'reliability': 88, 'wetPerformance': 86,
#         'straightSpeed': 82, 'aero': 86, 'powerUnit': 'Mercedes', 'championships': 0, 'image': '🏎'
#     },
#     'Red Bull RB6': {
#         'team': 'Red Bull Racing', 'year': 2010, 'downforce': 92, 'topSpeed': 318,
#         'acceleration': 87, 'cornering': 94, 'reliability': 90, 'wetPerformance': 88,
#         'straightSpeed': 80, 'aero': 94, 'powerUnit': 'Renault', 'championships': 1, 'image': '🏎'
#     },
#     'Red Bull RB7': {
#         'team': 'Red Bull Racing', 'year': 2011, 'downforce': 94, 'topSpeed': 322,
#         'acceleration': 89, 'cornering': 96, 'reliability': 91, 'wetPerformance': 90,
#         'straightSpeed': 81, 'aero': 96, 'powerUnit': 'Renault', 'championships': 1, 'image': '🏎'
#     },
#     'Mercedes W05': {
#         'team': 'Mercedes AMG', 'year': 2014, 'downforce': 90, 'topSpeed': 335,
#         'acceleration': 92, 'cornering': 93, 'reliability': 95, 'wetPerformance': 94,
#         'straightSpeed': 88, 'aero': 92, 'powerUnit': 'Mercedes', 'championships': 1, 'image': '🏎'
#     },
#     'Red Bull RB16B': {
#         'team': 'Red Bull Racing', 'year': 2021, 'downforce': 92, 'topSpeed': 343,
#         'acceleration': 91, 'cornering': 95, 'reliability': 93, 'wetPerformance': 91,
#         'straightSpeed': 85, 'aero': 94, 'powerUnit': 'Honda', 'championships': 1, 'image': '🏎'
#     },
#     'Mercedes W12': {
#         'team': 'Mercedes AMG', 'year': 2021, 'downforce': 91, 'topSpeed': 341,
#         'acceleration': 90, 'cornering': 94, 'reliability': 95, 'wetPerformance': 95,
#         'straightSpeed': 87, 'aero': 93, 'powerUnit': 'Mercedes', 'championships': 0, 'image': '🏎'
#     },
#     'Ferrari SF71H': {
#         'team': 'Scuderia Ferrari', 'year': 2018, 'downforce': 90, 'topSpeed': 349,
#         'acceleration': 92, 'cornering': 91, 'reliability': 86, 'wetPerformance': 88,
#         'straightSpeed': 94, 'aero': 91, 'powerUnit': 'Ferrari', 'championships': 0, 'image': '🏎'
#     },
#     'Mercedes W07': {
#         'team': 'Mercedes AMG', 'year': 2016, 'downforce': 92, 'topSpeed': 340,
#         'acceleration': 93, 'cornering': 95, 'reliability': 96, 'wetPerformance': 95,
#         'straightSpeed': 87, 'aero': 94, 'powerUnit': 'Mercedes', 'championships': 1, 'image': '🏎'
#     },
#     'Alpine A522': {
#         'team': 'Alpine F1', 'year': 2022, 'downforce': 86, 'topSpeed': 338,
#         'acceleration': 87, 'cornering': 88, 'reliability': 89, 'wetPerformance': 87,
#         'straightSpeed': 86, 'aero': 87, 'powerUnit': 'Renault', 'championships': 0, 'image': '🏎'
#     },
#     'Aston Martin AMR23': {
#         'team': 'Aston Martin', 'year': 2023, 'downforce': 90, 'topSpeed': 344,
#         'acceleration': 91, 'cornering': 92, 'reliability': 90, 'wetPerformance': 89,
#         'straightSpeed': 88, 'aero': 91, 'powerUnit': 'Mercedes', 'championships': 0, 'image': '🏎'
#     },
#     'Ferrari F2004': {
#         'team': 'Scuderia Ferrari', 'year': 2004, 'downforce': 88, 'topSpeed': 365,
#         'acceleration': 90, 'cornering': 91, 'reliability': 97, 'wetPerformance': 87,
#         'straightSpeed': 98, 'aero': 89, 'powerUnit': 'Ferrari', 'championships': 1, 'image': '🏎'
#     }
# }

# def calculate_score(car, conditions):
#     """Calculate performance score based on race conditions"""
#     score = 0
#     factors = []
    
#     # Weather impact
#     if conditions['weather'] in ['wet', 'rain']:
#         score += car['wetPerformance'] * 1.5
#         factors.append({'name': 'Wet Weather', 'value': car['wetPerformance'], 'weight': 1.5})
#     else:
#         score += car['topSpeed'] * 0.8
#         factors.append({'name': 'Dry Performance', 'value': car['topSpeed'] / 4, 'weight': 0.8})
    
#     # Track type impact
#     if conditions['trackType'] == 'street':
#         score += car['cornering'] * 1.3
#         score += car['acceleration'] * 1.2
#         factors.append({'name': 'Cornering', 'value': car['cornering'], 'weight': 1.3})
#     elif conditions['trackType'] == 'high-speed':
#         score += car['straightSpeed'] * 1.5
#         score += car['topSpeed'] * 0.01
#         factors.append({'name': 'Top Speed', 'value': car['topSpeed'] / 4, 'weight': 1.5})
#     else:
#         score += car['downforce'] * 1.2
#         factors.append({'name': 'Downforce', 'value': car['downforce'], 'weight': 1.2})
    
#     # Waviness impact
#     if conditions['waviness'] == 'bumpy':
#         score += car['reliability'] * 1.1
#         factors.append({'name': 'Reliability', 'value': car['reliability'], 'weight': 1.1})
    
#     # Curves impact
#     if conditions['curves'] == 'high':
#         score += car['cornering'] * 1.4
#         score += car['aero'] * 1.2
#     elif conditions['curves'] == 'low':
#         score += car['straightSpeed'] * 1.3
    
#     # Race duration impact
#     duration = int(conditions['raceDuration'])
#     if duration >= 12:
#         score += car['reliability'] * 1.5
#         factors.append({'name': 'Endurance', 'value': car['reliability'], 'weight': 1.5})
#     else:
#         score += car['acceleration'] * 1.2
    
#     return {'score': score, 'factors': factors}

# @app.route('/analyze', methods=['POST'])
# def analyze():
#     """Analyze race conditions and return recommendations"""
#     conditions = request.json
    
#     # Calculate scores for all cars
#     results = []
#     for name, car in F1_CARS.items():
#         calc_result = calculate_score(car, conditions)
#         results.append({
#             'name': name,
#             'car': car,
#             'score': calc_result['score'],
#             'factors': calc_result['factors']
#         })
    
#     # Sort by score
#     results.sort(key=lambda x: x['score'], reverse=True)
    
#     # Return top 5 and all results
#     return jsonify({
#         'recommended': results[0],
#         'alternatives': results[1:5],
#         'allCars': results
#     })
# @app.route('/focus_detector')
# def focus_detector():
#     return render_template('home1.html')

# @app.route('/index')
# def index():
#     return render_template('index.html')  
# if __name__ == '_main_':
#     app.run(debug=True, host='0.0.0.0', port=5000)


from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
import time
import winsound  # for beep alert (Windows only)
import json

app = Flask(__name__)

# ---------- SETTINGS ----------
MAR_YAWN_THRESHOLD = 0.6
EAR_BLINK_THRESHOLD = 0.23
EAR_SLEEP_THRESHOLD = 0.25  # Slightly increased for better sensitivity
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

# ---------- HELPER FUNCTIONS ----------
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

# ---------- DROWSINESS DETECTOR ----------
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

            # sleep logic (fixed for 2 seconds)
            if ear < EAR_SLEEP_THRESHOLD:
                if sleep_start_time is None:
                    sleep_start_time = time.time()
                elif (time.time() - sleep_start_time) >= SLEEP_SECONDS:
                    if not sleeping:
                        sleeping = True
                        winsound.Beep(ALERT_FREQ, ALERT_DUR)
            else:
                sleep_start_time = None
                sleeping = False

            # Draw visual feedback
            if sleeping:
                cv2.putText(frame, "⚠️ DROWSINESS ALERT! Stay Awake!", (60, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            else:
                cv2.putText(frame, "Status: Awake", (60, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            cv2.putText(frame, f"EAR: {ear:.2f}  MAR: {mar:.2f}", (20, h - 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Blinks: {blinks}  Yawns: {yawns}", (20, h - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Encode frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# ---------- ROUTES ----------
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ---------- F1 ANALYZER ----------
F1_CARS = {
    'Red Bull RB19': {'team': 'Red Bull Racing', 'year': 2023, 'downforce': 95, 'topSpeed': 350,
                      'acceleration': 94, 'cornering': 98, 'reliability': 96, 'wetPerformance': 92,
                      'straightSpeed': 88, 'aero': 98, 'powerUnit': 'Honda RBPT', 'championships': 1, 'image': '🏎'},
    'Mercedes W11': {'team': 'Mercedes AMG', 'year': 2020, 'downforce': 96, 'topSpeed': 348,
                     'acceleration': 95, 'cornering': 97, 'reliability': 98, 'wetPerformance': 96,
                     'straightSpeed': 89, 'aero': 97, 'powerUnit': 'Mercedes', 'championships': 1, 'image': '🏎'},
    'Ferrari F2004': {'team': 'Scuderia Ferrari', 'year': 2004, 'downforce': 88, 'topSpeed': 365,
                      'acceleration': 90, 'cornering': 91, 'reliability': 97, 'wetPerformance': 87,
                      'straightSpeed': 98, 'aero': 89, 'powerUnit': 'Ferrari', 'championships': 1, 'image': '🏎'}
}

def calculate_score(car, conditions):
    """Calculate performance score based on race conditions"""
    score = 0
    factors = []

    # Weather
    if conditions['weather'] in ['wet', 'rain']:
        score += car['wetPerformance'] * 1.5
        factors.append({'name': 'Wet Weather', 'value': car['wetPerformance'], 'weight': 1.5})
    else:
        score += car['topSpeed'] * 0.8
        factors.append({'name': 'Dry Performance', 'value': car['topSpeed'], 'weight': 0.8})

    # Track type
    if conditions['trackType'] == 'street':
        score += car['cornering'] * 1.3
        score += car['acceleration'] * 1.2
    elif conditions['trackType'] == 'high-speed':
        score += car['straightSpeed'] * 1.5
        score += car['topSpeed'] * 0.01
    else:
        score += car['downforce'] * 1.2

    # Waviness
    if conditions['waviness'] == 'bumpy':
        score += car['reliability'] * 1.1

    # Curves
    if conditions['curves'] == 'high':
        score += car['cornering'] * 1.4
        score += car['aero'] * 1.2
    elif conditions['curves'] == 'low':
        score += car['straightSpeed'] * 1.3

    # Race duration
    duration = int(conditions['raceDuration'])
    if duration >= 12:
        score += car['reliability'] * 1.5
    else:
        score += car['acceleration'] * 1.2

    return {'score': score, 'factors': factors}

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze race conditions and return recommendations"""
    conditions = request.json
    results = []
    for name, car in F1_CARS.items():
        calc_result = calculate_score(car, conditions)
        results.append({
            'name': name,
            'car': car,
            'score': calc_result['score'],
            'factors': calc_result['factors']
        })
    results.sort(key=lambda x: x['score'], reverse=True)
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

# ---------- MAIN ----------
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)


import cv2
import mediapipe as mp
import numpy as np
import time
import winsound
from collections import deque
from flask import Flask, Response, render_template, jsonify
import threading

app = Flask(__name__)

# ---------- IMPROVED THRESHOLDS ----------
# Eye Aspect Ratio thresholds
EAR_BLINK_THRESHOLD = 0.21      # Quick blink detection
EAR_DROWSY_THRESHOLD = 0.25     # Eyes partially closed (drowsy)
EAR_SLEEP_THRESHOLD = 0.18      # Eyes fully closed (sleeping)

# Mouth Aspect Ratio threshold
MAR_YAWN_THRESHOLD = 0.6        # Yawn detection

# Head pose thresholds
HEAD_TILT_THRESHOLD = 15        # degrees
HEAD_NOD_THRESHOLD = 20         # degrees

# Timing thresholds
DROWSY_TIME = 1.5               # seconds of low EAR before drowsy alert
SLEEP_TIME = 2.0                # seconds of very low EAR before sleep alert
CONSECUTIVE_FRAMES = 3          # frames to confirm detection
YAWN_DURATION = 2.0             # seconds for sustained yawn

# Alert settings
ALERT_FREQ = 2500               # Hz
ALERT_DUR = 1000                # milliseconds
ALERT_COOLDOWN = 5              # seconds between alerts

# Performance settings
FRAME_BUFFER_SIZE = 10          # frames for smoothing
EAR_SMOOTH_WINDOW = 5           # frames to average EAR

# ---------- MEDIAPIPE SETUP ----------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ---------- HELPER FUNCTIONS ----------
def euclidean(p1, p2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def eye_aspect_ratio(landmarks, w, h, side="left"):
    """
    Calculate Eye Aspect Ratio (EAR)
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    if side == "right":
        # Right eye landmarks
        ids = [33, 160, 158, 133, 153, 144]
    else:
        # Left eye landmarks
        ids = [362, 385, 387, 263, 380, 373]
    
    points = [(landmarks[i].x * w, landmarks[i].y * h) for i in ids]
    
    # Vertical distances
    vertical1 = euclidean(points[1], points[5])
    vertical2 = euclidean(points[2], points[4])
    
    # Horizontal distance
    horizontal = euclidean(points[0], points[3])
    
    # Avoid division by zero
    if horizontal < 0.001:
        return 0.0
    
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

def mouth_aspect_ratio(landmarks, w, h):
    """
    Calculate Mouth Aspect Ratio (MAR)
    MAR = ||upper-lower|| / ||left-right||
    """
    # Mouth landmarks
    upper = (landmarks[13].x * w, landmarks[13].y * h)
    lower = (landmarks[14].x * w, landmarks[14].y * h)
    left = (landmarks[61].x * w, landmarks[61].y * h)
    right = (landmarks[291].x * w, landmarks[291].y * h)
    
    vertical = euclidean(upper, lower)
    horizontal = euclidean(left, right)
    
    if horizontal < 0.001:
        return 0.0
    
    mar = vertical / horizontal
    return mar

def get_head_pose(landmarks, w, h):
    """
    Estimate head pose angles (pitch, yaw, roll)
    Returns angles in degrees
    """
    # Key facial landmarks for head pose estimation
    nose_tip = np.array([landmarks[1].x * w, landmarks[1].y * h])
    chin = np.array([landmarks[152].x * w, landmarks[152].y * h])
    left_eye = np.array([landmarks[33].x * w, landmarks[33].y * h])
    right_eye = np.array([landmarks[263].x * w, landmarks[263].y * h])
    left_mouth = np.array([landmarks[61].x * w, landmarks[61].y * h])
    right_mouth = np.array([landmarks[291].x * w, landmarks[291].y * h])
    
    # Calculate pitch (nodding up/down)
    face_height = euclidean(nose_tip, chin)
    eye_mouth_dist = euclidean((left_eye + right_eye) / 2, (left_mouth + right_mouth) / 2)
    pitch = np.arctan2(nose_tip[1] - chin[1], face_height) * 180 / np.pi
    
    # Calculate yaw (turning left/right)
    eye_center = (left_eye + right_eye) / 2
    nose_to_eye = nose_tip - eye_center
    yaw = np.arctan2(nose_to_eye[0], face_height) * 180 / np.pi
    
    # Calculate roll (tilting left/right)
    eye_diff = right_eye - left_eye
    roll = np.arctan2(eye_diff[1], eye_diff[0]) * 180 / np.pi
    
    return pitch, yaw, roll

def calculate_perclos(ear_buffer, threshold):
    """
    Calculate PERCLOS (Percentage of Eye Closure)
    Standard drowsiness metric used in research
    """
    if len(ear_buffer) == 0:
        return 0.0
    
    closed_count = sum(1 for ear in ear_buffer if ear < threshold)
    return (closed_count / len(ear_buffer)) * 100

# ---------- ENHANCED DROWSINESS DETECTOR ----------
class DrowsinessState:
    def __init__(self):
        self.yawns = 0
        self.blinks = 0
        self.microsleeps = 0
        self.drowsy_alerts = 0
        
        self.blink_active = False
        self.yawn_active = False
        self.yawn_start_time = None
        
        self.drowsy_start_time = None
        self.sleep_start_time = None
        self.last_alert_time = 0
        
        self.is_drowsy = False
        self.is_sleeping = False
        
        # Buffers for smoothing
        self.ear_buffer = deque(maxlen=FRAME_BUFFER_SIZE)
        self.mar_buffer = deque(maxlen=FRAME_BUFFER_SIZE)
        self.perclos_buffer = deque(maxlen=30)  # 30 frames for PERCLOS
        
        # Head pose tracking
        self.head_down_start = None
        self.head_tilt_start = None
        
        # Statistics
        self.start_time = time.time()
        self.face_detected_frames = 0
        self.total_frames = 0

state = DrowsinessState()
alert_lock = threading.Lock()

def play_alert_async():
    """Play alert sound in separate thread to avoid blocking"""
    with alert_lock:
        try:
            winsound.Beep(ALERT_FREQ, ALERT_DUR)
        except:
            pass  # Fail silently if sound not available

def generate_frames():
    """Generate video frames with drowsiness detection"""
    cap = cv2.VideoCapture(0)
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    fps_start_time = time.time()
    fps_frame_count = 0
    current_fps = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        state.total_frames += 1
        h, w = frame.shape[:2]
        
        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        
        # Calculate FPS
        fps_frame_count += 1
        if time.time() - fps_start_time > 1:
            current_fps = fps_frame_count
            fps_frame_count = 0
            fps_start_time = time.time()
        
        current_time = time.time()
        
        if results.multi_face_landmarks:
            state.face_detected_frames += 1
            face = results.multi_face_landmarks[0]
            landmarks = face.landmark
            
            # Calculate metrics
            left_ear = eye_aspect_ratio(landmarks, w, h, "left")
            right_ear = eye_aspect_ratio(landmarks, w, h, "right")
            ear = (left_ear + right_ear) / 2.0
            mar = mouth_aspect_ratio(landmarks, w, h)
            
            # Add to buffers for smoothing
            state.ear_buffer.append(ear)
            state.mar_buffer.append(mar)
            state.perclos_buffer.append(ear)
            
            # Smooth EAR using moving average
            if len(state.ear_buffer) >= EAR_SMOOTH_WINDOW:
                smooth_ear = np.mean(list(state.ear_buffer)[-EAR_SMOOTH_WINDOW:])
            else:
                smooth_ear = ear
            
            # Calculate PERCLOS
            perclos = calculate_perclos(state.perclos_buffer, EAR_DROWSY_THRESHOLD)
            
            # Get head pose
            pitch, yaw, roll = get_head_pose(landmarks, w, h)
            
            # ---------- BLINK DETECTION ----------
            if smooth_ear < EAR_BLINK_THRESHOLD:
                if not state.blink_active:
                    state.blink_active = True
            else:
                if state.blink_active:
                    state.blinks += 1
                    state.blink_active = False
            
            # ---------- YAWN DETECTION ----------
            if mar > MAR_YAWN_THRESHOLD:
                if not state.yawn_active:
                    state.yawn_start_time = current_time
                    state.yawn_active = True
                elif (current_time - state.yawn_start_time) >= YAWN_DURATION:
                    # Sustained yawn detected
                    if not state.is_drowsy:
                        state.is_drowsy = True
            else:
                if state.yawn_active:
                    state.yawns += 1
                state.yawn_active = False
                state.yawn_start_time = None
            
            # ---------- DROWSINESS DETECTION ----------
            if smooth_ear < EAR_DROWSY_THRESHOLD:
                if state.drowsy_start_time is None:
                    state.drowsy_start_time = current_time
                elif (current_time - state.drowsy_start_time) >= DROWSY_TIME:
                    if not state.is_drowsy:
                        state.is_drowsy = True
                        state.drowsy_alerts += 1
            else:
                state.drowsy_start_time = None
                if smooth_ear > EAR_DROWSY_THRESHOLD + 0.05:  # Hysteresis
                    state.is_drowsy = False
            
            # ---------- SLEEP/MICROSLEEP DETECTION ----------
            if smooth_ear < EAR_SLEEP_THRESHOLD:
                if state.sleep_start_time is None:
                    state.sleep_start_time = current_time
                elif (current_time - state.sleep_start_time) >= SLEEP_TIME:
                    if not state.is_sleeping:
                        state.is_sleeping = True
                        state.microsleeps += 1
                        
                        # Trigger alert with cooldown
                        if (current_time - state.last_alert_time) >= ALERT_COOLDOWN:
                            threading.Thread(target=play_alert_async, daemon=True).start()
                            state.last_alert_time = current_time
            else:
                state.sleep_start_time = None
                if smooth_ear > EAR_SLEEP_THRESHOLD + 0.05:  # Hysteresis
                    state.is_sleeping = False
            
            # ---------- HEAD POSE MONITORING ----------
            head_down = False
            head_tilted = False
            
            if pitch > HEAD_NOD_THRESHOLD:
                if state.head_down_start is None:
                    state.head_down_start = current_time
                elif (current_time - state.head_down_start) >= 2.0:
                    head_down = True
            else:
                state.head_down_start = None
            
            if abs(roll) > HEAD_TILT_THRESHOLD:
                if state.head_tilt_start is None:
                    state.head_tilt_start = current_time
                elif (current_time - state.head_tilt_start) >= 2.0:
                    head_tilted = True
            else:
                state.head_tilt_start = None
            
            # ---------- VISUAL FEEDBACK ----------
            # Status bar at top
            status_color = (0, 255, 0)  # Green
            status_text = "✓ Alert & Focused"
            
            if state.is_sleeping or head_down:
                status_color = (0, 0, 255)  # Red
                status_text = "⚠️ DANGER: SLEEPING DETECTED!"
            elif state.is_drowsy or perclos > 70 or head_tilted:
                status_color = (0, 165, 255)  # Orange
                status_text = "⚠ WARNING: Drowsiness Detected"
            elif perclos > 50:
                status_color = (0, 255, 255)  # Yellow
                status_text = "⚡ Attention: Fatigue Signs"
            
            # Draw status bar
            cv2.rectangle(frame, (0, 0), (w, 80), status_color, -1)
            cv2.putText(frame, status_text, (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            
            # Draw metrics panel
            panel_y = 100
            line_height = 30
            
            metrics = [
                f"EAR: {smooth_ear:.3f} (Blinks: {state.blinks})",
                f"MAR: {mar:.2f} (Yawns: {state.yawns})",
                f"PERCLOS: {perclos:.1f}%",
                f"Head: P{pitch:.0f}° Y{yaw:.0f}° R{roll:.0f}°",
                f"Microsleeps: {state.microsleeps}",
                f"FPS: {current_fps}"
            ]
            
            for i, metric in enumerate(metrics):
                y_pos = panel_y + i * line_height
                cv2.putText(frame, metric, (20, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw EAR indicator bar
            bar_x = w - 80
            bar_y = 100
            bar_height = 200
            bar_width = 40
            
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                         (50, 50, 50), -1)
            
            # Draw EAR level
            ear_normalized = min(1.0, max(0.0, smooth_ear / 0.4))
            fill_height = int(bar_height * ear_normalized)
            
            if smooth_ear < EAR_SLEEP_THRESHOLD:
                bar_color = (0, 0, 255)
            elif smooth_ear < EAR_DROWSY_THRESHOLD:
                bar_color = (0, 165, 255)
            else:
                bar_color = (0, 255, 0)
            
            cv2.rectangle(frame, (bar_x, bar_y + bar_height - fill_height),
                         (bar_x + bar_width, bar_y + bar_height), bar_color, -1)
            
            cv2.putText(frame, "EYE", (bar_x - 5, bar_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw threshold lines
            sleep_line_y = bar_y + int(bar_height * (1 - EAR_SLEEP_THRESHOLD / 0.4))
            drowsy_line_y = bar_y + int(bar_height * (1 - EAR_DROWSY_THRESHOLD / 0.4))
            
            cv2.line(frame, (bar_x - 5, sleep_line_y), (bar_x + bar_width + 5, sleep_line_y),
                    (0, 0, 255), 2)
            cv2.line(frame, (bar_x - 5, drowsy_line_y), (bar_x + bar_width + 5, drowsy_line_y),
                    (0, 165, 255), 2)
            
            # Warnings
            if head_down:
                cv2.putText(frame, "HEAD DOWN!", (w//2 - 100, h - 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            if head_tilted:
                cv2.putText(frame, "HEAD TILTED!", (w//2 - 120, h - 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)
        
        else:
            # No face detected
            cv2.putText(frame, "⚠ NO FACE DETECTED", (w//2 - 200, h//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            
            # Reset timers when no face detected
            state.drowsy_start_time = None
            state.sleep_start_time = None
            state.head_down_start = None
            state.head_tilt_start = None
        
        # Encode and yield frame
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()

# ---------- ROUTES ----------
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def get_stats():
    """Return current statistics"""
    uptime = time.time() - state.start_time
    detection_rate = (state.face_detected_frames / max(1, state.total_frames)) * 100
    
    return jsonify({
        'blinks': state.blinks,
        'yawns': state.yawns,
        'microsleeps': state.microsleeps,
        'drowsy_alerts': state.drowsy_alerts,
        'uptime': int(uptime),
        'detection_rate': round(detection_rate, 1),
        'is_drowsy': state.is_drowsy,
        'is_sleeping': state.is_sleeping
    })

@app.route('/focus_detector')
def focus_detector():
    return render_template('home1.html')

# ---------- F1 ANALYZER (keeping your existing code) ----------
F1_CARS = {
    'Red Bull RB19': {'team': 'Red Bull Racing', 'year': 2023, 'downforce': 95, 'topSpeed': 350,
                      'acceleration': 94, 'cornering': 98, 'reliability': 96, 'wetPerformance': 92,
                      'straightSpeed': 88, 'aero': 98, 'powerUnit': 'Honda RBPT', 'championships': 1, 'image': '🏎'},
    'Mercedes W11': {'team': 'Mercedes AMG', 'year': 2020, 'downforce': 96, 'topSpeed': 348,
                     'acceleration': 95, 'cornering': 97, 'reliability': 98, 'wetPerformance': 96,
                     'straightSpeed': 89, 'aero': 97, 'powerUnit': 'Mercedes', 'championships': 1, 'image': '🏎'},
    'Ferrari F2004': {'team': 'Scuderia Ferrari', 'year': 2004, 'downforce': 88, 'topSpeed': 365,
                      'acceleration': 90, 'cornering': 91, 'reliability': 97, 'wetPerformance': 87,
                      'straightSpeed': 98, 'aero': 89, 'powerUnit': 'Ferrari', 'championships': 1, 'image': '🏎'}
}

def calculate_score(car, conditions):
    score = 0
    factors = []
    if conditions['weather'] in ['wet', 'rain']:
        score += car['wetPerformance'] * 1.5
        factors.append({'name': 'Wet Weather', 'value': car['wetPerformance'], 'weight': 1.5})
    else:
        score += car['topSpeed'] * 0.8
        factors.append({'name': 'Dry Performance', 'value': car['topSpeed'], 'weight': 0.8})
    if conditions['trackType'] == 'street':
        score += car['cornering'] * 1.3
        score += car['acceleration'] * 1.2
    elif conditions['trackType'] == 'high-speed':
        score += car['straightSpeed'] * 1.5
        score += car['topSpeed'] * 0.01
    else:
        score += car['downforce'] * 1.2
    if conditions['waviness'] == 'bumpy':
        score += car['reliability'] * 1.1
    if conditions['curves'] == 'high':
        score += car['cornering'] * 1.4
        score += car['aero'] * 1.2
    elif conditions['curves'] == 'low':
        score += car['straightSpeed'] * 1.3
    duration = int(conditions['raceDuration'])
    if duration >= 12:
        score += car['reliability'] * 1.5
    else:
        score += car['acceleration'] * 1.2
    return {'score': score, 'factors': factors}

@app.route('/analyze', methods=['POST'])
def analyze():
    conditions = request.json
    results = []
    for name, car in F1_CARS.items():
        calc_result = calculate_score(car, conditions)
        results.append({
            'name': name,
            'car': car,
            'score': calc_result['score'],
            'factors': calc_result['factors']
        })
    results.sort(key=lambda x: x['score'], reverse=True)
    return jsonify({
        'recommended': results[0],
        'alternatives': results[1:5],
        'allCars': results
    })

@app.route('/index')
def index():
    return render_template('index.html')

# ---------- MAIN ----------
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

