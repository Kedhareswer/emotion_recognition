# Emotion Recognition System

A real-time emotion recognition system that can detect and classify facial emotions using deep learning. The system supports both web-based and desktop applications for emotion detection through webcam feed.

## Features

- Real-time emotion detection from webcam feed
- Support for 7 different emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
- Web interface for easy access
- Desktop application with a user-friendly GUI
- Pre-trained deep learning model for accurate emotion recognition

## Project Structure

```
├── app.py                      # Flask web application
├── emotion_recognition.py      # Core emotion recognition logic
├── emotion_recognition_ui.py   # Desktop UI implementation
├── models/                     # Pre-trained model files
├── static/                     # Static assets for web interface
├── templates/                  # HTML templates
└── webcam_emotion_recognition.py # Webcam-based emotion detection
```

## Prerequisites

- Python 3.8+
- Webcam (for real-time detection)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Kedhareswer/emotion-recognition.git
cd emotion-recognition
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Web Application

1. Start the Flask server:
```bash
python app.py
```
2. Open your web browser and navigate to `http://localhost:5000`
3. Grant camera permissions when prompted
4. The system will start detecting emotions in real-time

### Desktop Application

1. Run the desktop application:
```bash
python emotion_recognition_ui.py
```
2. The GUI will open with your webcam feed
3. Emotion detection will start automatically

## Model Information

The system uses a deep learning model trained on the FER2013 dataset. The model architecture is based on a Convolutional Neural Network (CNN) optimized for facial emotion recognition.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.