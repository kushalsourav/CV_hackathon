from flask import Flask, render_template, Response, jsonify

import threading
import cv2
import mediapipe as mp
import numpy as np
import time
from gtts import gTTS
import pygame
import os

# Mediapipe initialization
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

is_mask_enabled = False
# Thresholds for drowsiness detection
EYE_AR_THRESH = 0.25
YAWN_THRESH = 0.6
DURATION_THRESH = 1.5

# Timers and flags
eye_closed_start_time = None
mouth_open_start_time = None
is_running = False
lock = threading.Lock()

# Flask app
app = Flask(__name__)

# Function to calculate Eye Aspect Ratio (EAR)
def calculate_ear(landmarks, indices):
    left = np.array(landmarks[indices[0]])
    right = np.array(landmarks[indices[3]])
    top = (np.array(landmarks[indices[1]]) + np.array(landmarks[indices[2]])) / 2
    bottom = (np.array(landmarks[indices[4]]) + np.array(landmarks[indices[5]])) / 2
    horizontal_dist = np.linalg.norm(left - right)
    vertical_dist = np.linalg.norm(top - bottom)
    return vertical_dist / horizontal_dist

# Function to calculate Mouth Aspect Ratio (MAR)
def calculate_mar(landmarks, indices):
    left = np.array(landmarks[indices[0]])
    right = np.array(landmarks[indices[6]])
    top = (np.array(landmarks[indices[3]]) + np.array(landmarks[indices[2]])) / 2
    bottom = (np.array(landmarks[indices[7]]) + np.array(landmarks[indices[8]])) / 2
    horizontal_dist = np.linalg.norm(left - right)
    vertical_dist = np.linalg.norm(top - bottom)
    return vertical_dist / horizontal_dist

# Function for voice alert using pygame
def voice_alert(message):
    tts = gTTS(message)
    tts.save("alert.mp3")
    pygame.mixer.init()
    pygame.mixer.music.load("alert.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        continue  # Wait for the audio to finish
    pygame.mixer.music.stop()
    pygame.mixer.quit()
    os.remove("alert.mp3")

def generate_frames():
    global is_running, eye_closed_start_time, mouth_open_start_time, is_mask_enabled
    cap = None

    # Initialize Mediapipe FaceMesh once
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    while True:
        with lock:
            if not is_running:
                if cap:
                    cap.release()
                    cap = None
                time.sleep(0.1)
                continue

            if cap is None:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    print("Error: Could not open camera")
                    break

        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw the face mesh if the mask is enabled
                if is_mask_enabled:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                    )

                # EAR and MAR calculations
                h, w, _ = frame.shape
                landmarks = [
                    (int(l.x * w), int(l.y * h)) for l in face_landmarks.landmark
                ]
                left_eye_indices = [362, 385, 387, 263, 373, 380]
                right_eye_indices = [33, 160, 158, 133, 153, 144]
                mouth_indices = [78, 308, 13, 312, 14, 82, 87, 317, 14]

                left_ear = calculate_ear(landmarks, left_eye_indices)
                right_ear = calculate_ear(landmarks, right_eye_indices)
                ear = (left_ear + right_ear) / 2.0
                mar = calculate_mar(landmarks, mouth_indices)

                # Drowsiness detection
                if ear < EYE_AR_THRESH:
                    if eye_closed_start_time is None:
                        eye_closed_start_time = time.time()
                    elif time.time() - eye_closed_start_time > DURATION_THRESH:
                        cv2.putText(frame, "DROWSINESS ALERT!", (100, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                        voice_alert("Drowsiness detected! Please take a break.")
                        eye_closed_start_time = None
                else:
                    eye_closed_start_time = None

                # Yawning detection
                if mar > YAWN_THRESH:
                    if mouth_open_start_time is None:
                        mouth_open_start_time = time.time()
                    elif time.time() - mouth_open_start_time > DURATION_THRESH:
                        cv2.putText(frame, "YAWNING ALERT!", (100, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                        voice_alert("Yawning detected! Stay alert.")
                        mouth_open_start_time = None
                else:
                    mouth_open_start_time = None

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    if cap:
        cap.release()
@app.route('/toggle_mask', methods=['POST'])
def toggle_mask():
    global is_mask_enabled
    is_mask_enabled = not is_mask_enabled  # Toggle the flag
    return jsonify({"mask_enabled": is_mask_enabled})


# Flask routes

alert_message = None

# Function to send an alert message
def send_alert(message):
    global alert_message
    alert_message = message

# Update the voice_alert function to also send the alert to the webpage
def voice_alert(message):
    global alert_message
    send_alert(message)  # Update the global alert message
    tts = gTTS(message)
    tts.save("alert.mp3")
    pygame.mixer.init()
    pygame.mixer.music.load("alert.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        continue  # Wait for the audio to finish
    pygame.mixer.music.stop()
    pygame.mixer.quit()
    os.remove("alert.mp3")

# Create a route for fetching alerts
@app.route('/alerts')
def alerts():
    global alert_message
    message = alert_message
    alert_message = None  # Reset after fetching
    return jsonify({"message": message})
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start', methods=['POST'])
def start():
    global is_running
    with lock:
        is_running = True
    return '', 200

@app.route('/stop', methods=['POST'])
def stop():
    global is_running
    with lock:
        is_running = False
    return '', 200


if __name__ == '__main__':
    app.run(debug=True)
