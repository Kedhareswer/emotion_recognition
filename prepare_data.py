import os
import requests
import zipfile
import shutil
from emotion_recognition import EMOTION_CLASSES

def download_dataset():
    # URL for the FER2013 dataset (you'll need to provide the actual URL)
    dataset_url = "https://www.kaggle.com/datasets/msambare/fer2013"
    print(f"Please download the FER2013 dataset from: {dataset_url}")
    print("After downloading, place the dataset in the 'data' directory and run this script again.")

def organize_dataset():
    data_dir = 'data'
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'validation')

    # Create directories for each emotion class
    for emotion in EMOTION_CLASSES:
        os.makedirs(os.path.join(train_dir, emotion), exist_ok=True)
        os.makedirs(os.path.join(val_dir, emotion), exist_ok=True)

    print("Created directory structure for the dataset.")
    print("Please organize your images into the following structure:")
    print("data/")
    print("  ├── train/")
    for emotion in EMOTION_CLASSES:
        print(f"  │   ├── {emotion}/")
    print("  ├── validation/")
    for emotion in EMOTION_CLASSES:
        print(f"  │   ├── {emotion}/")

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    
    # Download and organize the dataset
    download_dataset()
    organize_dataset()
    
    print("\nOnce you have organized the dataset, you can run train_model.py to train the emotion recognition model.")
