from flask import Flask, Response, render_template, jsonify
import cv2
import numpy as np
import tensorflow as tf
from collections import deque
from PIL import Image
import io
import base64

app = Flask(__name__)

# Configure TensorFlow for memory efficiency
tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Define the emotion classes
EMOTION_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Load the trained model
try:
    model = tf.keras.models.load_model('models/model_fer2013.h5')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    tf.config.run_functions_eagerly(False)
    model.make_predict_function()
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Initialize face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize emotion prediction smoothing
emotion_buffer = deque(maxlen=10)
confidence_threshold = 0.35

def detect_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(blurred)
    faces = face_cascade.detectMultiScale(enhanced, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        padding = int(0.1 * w)
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(frame.shape[1] - x, w + 2 * padding)
        h = min(frame.shape[0] - y, h + 2 * padding)
        face_roi = frame[y:y+h, x:x+w]
        return face_roi, (x, y, w, h)
    return None, None

def preprocess_frame(frame):
    if frame is None:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    resized = cv2.resize(blurred, (48, 48), interpolation=cv2.INTER_LANCZOS4)
    normalized = resized.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=(0, -1))

def get_smoothed_prediction(predictions):
    if predictions is None:
        return None, None, 0.0
        
    emotion_idx = np.argmax(predictions[0])
    confidence = predictions[0][emotion_idx]
    
    if confidence < confidence_threshold:
        return None, None, 0.0
        
    emotion_buffer.append(emotion_idx)
    emotion_counts = np.bincount(list(emotion_buffer))
    smoothed_emotion_idx = np.argmax(emotion_counts)
    
    return smoothed_emotion_idx, EMOTION_CLASSES[smoothed_emotion_idx], confidence

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        # Get frame data from the request
        frame_data = request.get_json()['frame']
        frame_bytes = base64.b64decode(frame_data.split(',')[1])
        frame_arr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(frame_arr, cv2.IMREAD_COLOR)

        # Process frame
        face_roi, face_coords = detect_face(frame)
        result = {
            'emotion': None,
            'confidence': 0,
            'face_detected': False
        }

        if face_roi is not None and model is not None:
            processed_face = preprocess_frame(face_roi)
            if processed_face is not None:
                predictions = model.predict(processed_face, verbose=0)
                emotion_idx, emotion_label, confidence = get_smoothed_prediction(predictions)
                
                if emotion_label is not None:
                    result = {
                        'emotion': emotion_label,
                        'confidence': float(confidence),
                        'face_detected': True,
                        'face_coords': {
                            'x': int(face_coords[0]),
                            'y': int(face_coords[1]),
                            'width': int(face_coords[2]),
                            'height': int(face_coords[3])
                        }
                    }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)