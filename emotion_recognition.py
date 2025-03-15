import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the emotion classes
EMOTION_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def prepare_data():
    # Create data generators with augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Validation data generator (only rescaling)
    val_datagen = ImageDataGenerator(rescale=1./255)

    # Set up data generators
    batch_size = 32
    train_generator = train_datagen.flow_from_directory(
        '/kaggle/input/fer2013/train',
        target_size=(224, 224),
        color_mode='rgb',
        batch_size=batch_size,
        class_mode='categorical',
        classes=EMOTION_CLASSES
    )

    # Assuming validation data is in the same directory
    validation_generator = val_datagen.flow_from_directory(
        '/kaggle/input/fer2013/test',
        target_size=(224, 224),
        color_mode='rgb',
        batch_size=batch_size,
        class_mode='categorical',
        classes=EMOTION_CLASSES
    )

    return train_generator, validation_generator

def create_model():
    # Create the model architecture
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(len(EMOTION_CLASSES), activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model

def train_emotion_model():
    # Create and compile the model
    model = create_model()
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Prepare the data
    train_generator, validation_generator = prepare_data()

    # Train the model
    epochs = 50
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(
                '/kaggle/working/models/emotion_model.keras',
                save_best_only=True,
                monitor='val_accuracy'
            )
        ]
    )

    return model, history

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('/kaggle/working/models', exist_ok=True)

    # Train the model
    model, history = train_emotion_model()
    
    # Save the final model
    model.save('/kaggle/working/models/emotion_model_final.keras')