#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os

def create_peak_detector(input_length=1024, num_peaks=5):
    model = models.Sequential([
        layers.Input(shape=(input_length, 1)),
        layers.Conv1D(32, kernel_size=5, padding='same', activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(64, kernel_size=5, padding='same', activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(128, kernel_size=5, padding='same', activation='relu'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dense(2 * num_peaks, activation='linear', name="output_layer")
    ])
    return model

if __name__ == "__main__":
    # Load data
    data = np.load('data/training_data.npz')
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']

    # Add channel dimension
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]

    num_peaks = y_train.shape[1] // 2
    signal_length = X_train.shape[1]

    model = create_peak_detector(input_length=signal_length, num_peaks=num_peaks)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.summary()

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32)

    # Save model as a SavedModel
    saved_model_dir = "saved_model_dir"
    if not os.path.exists(saved_model_dir):
        os.makedirs(saved_model_dir)
    tf.saved_model.save(model, saved_model_dir)
    print("Model saved to SavedModel format.")

