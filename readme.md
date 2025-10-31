💤 Drowsiness Detection Web App (Flask + Mediapipe)

This project detects driver drowsiness and yawns in real time using a webcam feed.
It uses Mediapipe Face Mesh for landmark detection and Flask to serve a web interface.
When a person closes their eyes for more than 2 seconds, a drowsiness alert is triggered.
It also counts yawns, blinks, and monitors the person’s distance from the camera to ensure they are visible and close enough.

🚀 Features

✅ Real-time face tracking using Mediapipe
✅ Eye Aspect Ratio (EAR)–based drowsiness detection
✅ Mouth Aspect Ratio (MAR)–based yawn detection
✅ Blink counter and yawn counter
✅ Distance check – alerts if user is too far from the camera
✅ Drowsiness alert (sound + red warning text) if eyes remain closed for more than 2 seconds
✅ Flask web interface – view the live camera feed from a browser
✅ Works on Windows, Linux, and macOS (Windows plays a beep sound)

🧩 Technologies Used

Flask – Web framework for live video feed

Mediapipe – Face mesh and landmarks detection

OpenCV – Video capture and image processing

Numpy – Numerical computations

Threading – For smooth real-time processing

Winsound – Beep sound alert (on Windows)