import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
from collections import deque

# Configure TensorFlow for memory efficiency
tf.config.set_visible_devices([], 'GPU')  # Use CPU only if memory is constrained
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Define the emotion classes (same as training)
EMOTION_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Load the trained model with memory-efficient configuration
try:
    # Create a lightweight model with reduced parameters
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(len(EMOTION_CLASSES), activation='softmax')
    ])
    try:
        # Load the model with pre-trained weights
        model = tf.keras.models.load_model('models/model_fer2013.h5')
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        tf.config.run_functions_eagerly(False)
        model.make_predict_function()
    except Exception as load_error:
        print(f"Error loading pre-trained model: {load_error}")
        # If loading fails, compile the lightweight model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print("Using lightweight model without pre-trained weights")

except Exception as e:
    print(f"Error loading model: {e}")
    model = None

class EmotionRecognitionApp:
    def __init__(self, window):
        self.window = window
        self.window.title('Emotion Recognition System')
        
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize emotion prediction smoothing with longer buffer and lower threshold
        self.emotion_buffer = deque(maxlen=10)
        self.confidence_threshold = 0.35
        
        # Create main container
        self.main_container = ttk.Frame(window)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create webcam frame
        self.webcam_frame = ttk.LabelFrame(self.main_container, text='Webcam Feed')
        self.webcam_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.webcam_label = ttk.Label(self.webcam_frame)
        self.webcam_label.pack()
        
        # Create sample images frame
        self.samples_frame = ttk.LabelFrame(self.main_container, text='Sample Images')
        self.samples_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create grid for sample images
        self.create_sample_grid()
        
        # Initialize webcam with error handling
        self.cap = None
        self.init_webcam()
        
        # Start webcam update
        self.update_webcam()
    
    def create_sample_grid(self):
        # Create a frame for each emotion with sample image and prediction
        for i, emotion in enumerate(EMOTION_CLASSES):
            frame = ttk.Frame(self.samples_frame)
            frame.grid(row=i//2, column=i%2, padx=5, pady=5)
            
            # Load sample image from data/train directory
            sample_path = f'data/train/{emotion}'
            if os.path.exists(sample_path) and os.listdir(sample_path):
                img_path = os.path.join(sample_path, os.listdir(sample_path)[0])
                img = Image.open(img_path)
                img = img.resize((100, 100))
                photo = ImageTk.PhotoImage(img)
                
                # Create image label
                img_label = ttk.Label(frame, image=photo)
                img_label.image = photo  # Keep a reference
                img_label.pack()
                
                # Create prediction label
                pred_label = ttk.Label(frame, text=f'{emotion}: --')
                pred_label.pack()
                
                # Bind click event
                img_label.bind('<Button-1>', 
                              lambda e, img=img_path, label=pred_label: 
                              self.analyze_sample(img, label))
    
    def detect_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(blurred)
        # Adjust face detection parameters for better detection
        faces = self.face_cascade.detectMultiScale(enhanced, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        
        if len(faces) > 0:
            (x, y, w, h) = faces[0]  # Get the first face
            # Add padding to include more context around the face
            padding = int(0.1 * w)
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(frame.shape[1] - x, w + 2 * padding)
            h = min(frame.shape[0] - y, h + 2 * padding)
            face_roi = frame[y:y+h, x:x+w]
            return face_roi, (x, y, w, h)
        return None, None
    
    def preprocess_frame(self, frame):
        if frame is None:
            return None
        # Enhanced preprocessing pipeline
        # Convert to grayscale and apply CLAHE for better contrast
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        # Resize to match model input size (48x48) using LANCZOS
        resized = cv2.resize(blurred, (48, 48), interpolation=cv2.INTER_LANCZOS4)
        # Normalize pixel values and add batch dimension
        normalized = resized.astype(np.float32) / 255.0
        # Reshape to match model's expected input shape (None, 48, 48, 1)
        return np.expand_dims(normalized, axis=(0, -1))
        
    def get_smoothed_prediction(self, predictions):
        if predictions is None:
            return None, None, 0.0
            
        emotion_idx = np.argmax(predictions[0])
        confidence = predictions[0][emotion_idx]
        
        if confidence < self.confidence_threshold:
            return None, None, 0.0
            
        self.emotion_buffer.append(emotion_idx)
        # Get most common emotion from buffer
        emotion_counts = np.bincount(list(self.emotion_buffer))
        smoothed_emotion_idx = np.argmax(emotion_counts)
        
        return smoothed_emotion_idx, EMOTION_CLASSES[smoothed_emotion_idx], confidence
    
    def analyze_sample(self, img_path, label):
        # Load and preprocess image
        img = cv2.imread(img_path)
        processed = self.preprocess_frame(img)
        
        # Get prediction
        predictions = model.predict(processed, verbose=0)
        emotion_idx = np.argmax(predictions[0])
        emotion_label = EMOTION_CLASSES[emotion_idx]
        confidence = predictions[0][emotion_idx]
        
        # Update label
        label.configure(text=f'{emotion_label}: {confidence:.2f}')
    
    def init_webcam(self):
        """Initialize webcam with retries and error handling"""
        max_retries = 3
        current_try = 0
        
        while current_try < max_retries:
            if self.cap is not None and self.cap.isOpened():
                self.cap.release()
            
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                # Test if we can actually read from the camera
                ret, _ = self.cap.read()
                if ret:
                    return True
            
            current_try += 1
            self.window.after(1000)  # Wait 1 second before retrying
        
        return False
    
    def update_webcam(self):
        if self.cap is None or not self.cap.isOpened():
            if not self.init_webcam():
                # Show error message in the webcam frame
                error_img = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(error_img, 'Camera not available', (50, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                img = Image.fromarray(cv2.cvtColor(error_img, cv2.COLOR_BGR2RGB))
                photo = ImageTk.PhotoImage(img)
                self.webcam_label.configure(image=photo)
                self.webcam_label.image = photo
                self.window.after(1000, self.update_webcam)  # Retry after 1 second
                return
        
        ret, frame = self.cap.read()
        if ret:
            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_roi, face_coords = self.detect_face(frame)
            
            if face_roi is not None and model is not None:
                try:
                    # Process detected face
                    processed_face = self.preprocess_frame(face_roi)
                    if processed_face is not None:
                        predictions = model.predict(processed_face, verbose=0)
                        emotion_idx, emotion_label, confidence = self.get_smoothed_prediction(predictions)
                        
                        if emotion_label is not None:
                            # Draw face rectangle and emotion text
                            (x, y, w, h) = face_coords
                            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            text = f'{emotion_label}: {confidence:.2f}'
                            cv2.putText(display_frame, text, (x, y-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                except Exception as e:
                    print(f"Error during prediction: {e}")
                    # Continue showing video feed even if prediction fails
                else:
                    cv2.putText(display_frame, 'Low confidence', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(display_frame, 'No face detected', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Convert to PhotoImage
            img = Image.fromarray(display_frame)
            img = img.resize((640, 480))
            photo = ImageTk.PhotoImage(img)
            
            # Update label
            self.webcam_label.configure(image=photo)
            self.webcam_label.image = photo
        
        # Schedule next update
        self.window.after(10, self.update_webcam)
    
    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

def main():
    root = tk.Tk()
    app = EmotionRecognitionApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()