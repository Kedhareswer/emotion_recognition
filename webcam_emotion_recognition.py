import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('models/emotion_model_final.keras')

# Define the emotion classes (same as training)
EMOTION_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def preprocess_frame(frame):
    # Resize and preprocess the frame
    resized = cv2.resize(frame, (224, 224))
    # Convert to RGB (OpenCV uses BGR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # Normalize pixel values
    normalized = rgb.astype(np.float32) / 255.0
    # Add batch dimension
    return np.expand_dims(normalized, axis=0)

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Make a copy for display
        display_frame = frame.copy()
        
        # Preprocess the frame
        processed_frame = preprocess_frame(frame)
        
        # Get prediction
        predictions = model.predict(processed_frame, verbose=0)
        emotion_idx = np.argmax(predictions[0])
        emotion_label = EMOTION_CLASSES[emotion_idx]
        confidence = predictions[0][emotion_idx]
        
        # Display results
        text = f'{emotion_label}: {confidence:.2f}'
        cv2.putText(display_frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show the frame
        cv2.imshow('Emotion Recognition', display_frame)
        
        # Break loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()