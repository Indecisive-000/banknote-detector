# Banknote Authenticity Detector
# Learning ML project
# Author: Platon

import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Classes
CLASSES = ['real', 'fake']
IMG_SIZE = (224, 224)

def load_and_preprocess_image(path):
    """Load image, resize and normalize."""
    img = Image.open(path).convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    return img_array

def prepare_dataset(data_dir):
    """Load all images from real/ and fake/ folders."""
    images = []
    labels = []
    for idx, class_name in enumerate(CLASSES):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: folder {class_dir} not found")
            continue
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_dir, img_name)
                try:
                    img = load_and_preprocess_image(img_path)
                    images.append(img)
                    labels.append(idx)
                except Exception as e:
                    print(f"Could not load {img_path}: {e}")
    return np.array(images), np.array(labels)

def build_model():
    """Simple Convolutional Neural Network."""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(CLASSES), activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_and_evaluate(data_dir='dataset', epochs=15):
    """Train the model and print metrics."""
    print("Loading images...")
    X, y = prepare_dataset(data_dir)
    if len(X) == 0:
        print("No images found. Create dataset/real and dataset/fake folders.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_model()
    model.summary()

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        validation_split=0.2,
        batch_size=16
    )

    # Predictions and metrics
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"\n=== Results ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASSES))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    model.save('banknote_detector_model.h5')
    print("\nModel saved as banknote_detector_model.h5")
    return model, history

def predict_single(model_path, image_path):
    """Predict one image."""
    model = tf.keras.models.load_model(model_path)
    img = load_and_preprocess_image(image_path)
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    class_idx = np.argmax(pred)
    confidence = float(pred[0][class_idx])
    return CLASSES[class_idx], confidence

if __name__ == "__main__":
    # Uncomment to train (after adding images to dataset/)
    # train_and_evaluate()

    # Example of prediction:
    # result, conf = predict_single('banknote_detector_model.h5', 'test.jpg')
    # print(f"Prediction: {result} (confidence: {conf:.2f})")
    print("Script is ready. Add images and uncomment the training line.")
