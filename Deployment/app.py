import cv2
import numpy as np
from flask import Flask, render_template, Response
import joblib

app = Flask(__name__)

IMG_SIZE = (48, 48)

EMOTION_MAP = {
    1: 'Disgust',
    3: 'Happiness',
    5: 'Surprise',
    6: 'Neutral'
}

print("Loading models...")
model = joblib.load('fer_model.pkl')
scaler = joblib.load('scaler.pkl')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
print("Models loaded!")

def predict_emotion(face_image):
    try:
        roi_resized = cv2.resize(face_image, IMG_SIZE)
        
        roi_flat = roi_resized.flatten().reshape(1, -1)
        
        roi_scaled = scaler.transform(roi_flat)
        
        prediction_idx = model.predict(roi_scaled)[0]
        label = EMOTION_MAP.get(prediction_idx, "Unknown")
        return label
    except Exception as e:
        return "Error"

def generate_frames():
    camera = cv2.VideoCapture(0) 
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        height, width = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        display_frame = cv2.flip(frame, 1)

        for (x, y, w, h) in faces:
        
            roi_gray = gray[y:y+h, x:x+w]
            
            emotion = predict_emotion(roi_gray)
            
            mirrored_x = width - x - w
            
            cv2.rectangle(display_frame, (mirrored_x, y), (mirrored_x+w, y+h), (255, 0, 0), 2)
            cv2.putText(display_frame, emotion, (mirrored_x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        ret, buffer = cv2.imencode('.jpg', display_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)