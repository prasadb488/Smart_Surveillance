from ultralytics import YOLO
import cv2

model = YOLO("/Users/saiprasadbanda/Documents/Smart_Surveillance/app/best.pt")  

def detect_emotion(frame):
    results = model(frame, verbose=False)

    annotated_frame = results[0].plot().copy()  
    emotions = []

    boxes = results[0].boxes
    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls[0])
            label = results[0].names[cls_id]
            emotions.append(label)

    return annotated_frame, emotions

