# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
# Import necessary libraries

import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
# Function to load audio data and extract MFCC features

data_dir = '/content/drive/MyDrive/SoundAudio'


def load_audio_data(data_dir):
    X = []
    y = []
    labels = os.listdir(data_dir)

    for label in labels:
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for file_name in os.listdir(label_dir):
            if not file_name.endswith(".wav"):
                continue
            file_path = os.path.join(label_dir, file_name)
            try:
                # Load the audio file
                audio, sample_rate = librosa.load(file_path, sr=None)
                # Extract MFCC features
                mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
                mfccs = np.mean(mfccs.T, axis=0)  # Mean of MFCC across time
                X.append(mfccs)
                y.append(label)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    return np.array(X), np.array(y)
# Function to extract MFCC from audio signal
def extract_mfcc_from_audio(audio, sample_rate, n_mfcc=13):
    mfccs = librosa.feature.mfcc(y=audio.flatten(), sr=sample_rate, n_mfcc=n_mfcc)
    mfccs = np.mean(mfccs.T, axis=0)
    return mfccs.reshape(1, -1)

# Function to predict digit from file
def predict_digit_from_file(model, le, file_path):
    audio, sample_rate = librosa.load(file_path, sr=None)
    mfccs = extract_mfcc_from_audio(audio, sample_rate)
    prediction = model.predict(mfccs)
    predicted_label = np.argmax(prediction)

    if predicted_label >= len(le.classes_):
        raise ValueError(f"Predicted label {predicted_label} is out of bounds.")

    print(f"Predicted label from file: {le.inverse_transform([predicted_label])[0]}")
    return predicted_label

sample_audio_file = '/content/drive/MyDrive/Sample/sample2.wav'
model_save_path = '/content/drive/MyDrive/speech_model.h5'

# Load audio dataset
X, y = load_audio_data(data_dir)

# Encode string labels into integers
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Ensure labels are in the correct range
if not np.all(np.isin(y_encoded, np.arange(len(le.classes_)))):
    raise ValueError("Encoded labels out of bounds.")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Define ANN model
model = models.Sequential([
    layers.InputLayer(input_shape=(13,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(len(le.classes_), activation='softmax')
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), batch_size=16)

# Evaluate on test data
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.2f}")

# Save the model
model.save(model_save_path)

# Plot training history
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure()
    plt.plot(epochs, acc, 'bo-', label='Training accuracy')
    plt.plot(epochs, val_acc, 'ro-', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(epochs, loss, 'bo-', label='Training loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

plot_history(history)
# Predict digit from a sample audio file
predicted_digit = predict_digit_from_file(model, le, sample_audio_file)
