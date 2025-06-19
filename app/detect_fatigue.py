
import cv2
import mediapipe as mp
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)


RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]

def get_ear(eye_landmarks):
    from math import dist
    A = dist(eye_landmarks[1], eye_landmarks[5])
    B = dist(eye_landmarks[2], eye_landmarks[4])
    C = dist(eye_landmarks[0], eye_landmarks[3])
    ear = (A + B) / (2.0 * C)
    return ear

def detect_fatigue(frame, results):
    fatigue = False
    h, w, _ = frame.shape
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            right_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in RIGHT_EYE]
            left_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in LEFT_EYE]
            right_ear = get_ear(right_eye)
            left_ear = get_ear(left_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < 0.22:
                fatigue = True
    return fatigue
