import cv2
import numpy as np
import mediapipe as mp
import time
import winsound  # for beep sound (Windows only)

# ---------- SETTINGS ----------
MAR_YAWN_THRESHOLD = 0.6     # Mouth Aspect Ratio threshold
EAR_BLINK_THRESHOLD = 0.23   # EAR below this => blink
EAR_SLEEP_THRESHOLD = 0.23   # <<<--- FIX: Set this to be the same as the blink threshold
SLEEP_SECONDS = 2.0          # Eyes closed for this many seconds => sleep
ALERT_FREQ = 1200            # Beep frequency
ALERT_DUR = 800              # Beep duration (ms)
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

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera.")
        return

    yawns = 0
    blinks = 0
    blink_active = False
    yawn_active = False
    sleep_start_time = None
    sleeping = False

    print("🟢 Starting Yawn, Blink, Sleep Detector — Press 'q' to quit")

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

            # EAR for both eyes
            left_ear = eye_aspect_ratio(landmarks, w, h, "left")
            right_ear = eye_aspect_ratio(landmarks, w, h, "right")
            ear = (left_ear + right_ear) / 2

            # MAR for mouth
            mar = mouth_aspect_ratio(landmarks, w, h)

            # ---------- BLINK DETECTION ----------
            if ear < EAR_BLINK_THRESHOLD:
                if not blink_active:
                    blink_active = True
            else:
                if blink_active:
                    blinks += 1
                    blink_active = False

            # ---------- YAWN DETECTION ----------
            if mar > MAR_YAWN_THRESHOLD:
                if not yawn_active:
                    yawns += 1
                    yawn_active = True
            else:
                yawn_active = False

            # ---------- SLEEP DETECTION ----------
            if ear < EAR_SLEEP_THRESHOLD:
                if sleep_start_time is None:
                    sleep_start_time = time.time()
                elif (time.time() - sleep_start_time) >= SLEEP_SECONDS:
                    sleeping = True
            else:
                sleep_start_time = None
                sleeping = False

            # ---------- ALERT WHEN SLEEPING ----------
            if sleeping:
                cv2.putText(frame, "⚠️ DROWSINESS ALERT!", (60, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
                winsound.Beep(ALERT_FREQ, ALERT_DUR)
            else:
                cv2.putText(frame, "Status: Awake", (60, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)

            # ---------- DISPLAY METRICS ----------
            cv2.putText(frame, f"EAR: {ear:.2f}  MAR: {mar:.2f}", (20, h - 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Blinks: {blinks}  Yawns: {yawns}", (20, h - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Yawn & Sleep Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()

if __name__ == "__main__":
    main()