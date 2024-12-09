#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import Sequence
import math

# --- Signal Generator ---
def generate_signal(num_peaks, signal_length=1024, noise_level=0.05, asymmetry_factor=0.2):
    x = np.linspace(0, signal_length, signal_length)
    signal = np.zeros_like(x)
    peak_positions = np.random.randint(50, signal_length - 50, num_peaks)
    fwhm_values = np.random.uniform(10, 50, num_peaks)
    amplitudes = np.random.uniform(0.5, 2.0, num_peaks)

    for pos, fwhm, amp in zip(peak_positions, fwhm_values, amplitudes):
        stddev = fwhm / (2 * np.sqrt(2 * np.log(2)))
        asymmetry = np.random.uniform(-asymmetry_factor, asymmetry_factor)
        gaussian = amp * np.exp(-((x - pos) ** 2) / (2 * stddev ** 2))
        asymmetric = gaussian * (1 + asymmetry * (x - pos) / fwhm)
        signal += np.clip(asymmetric, 0, None)

    noise = np.random.normal(0, noise_level * np.max(signal), signal_length)
    signal += noise

    idx = np.argsort(peak_positions)
    return signal, np.hstack([peak_positions[idx], fwhm_values[idx]])

# --- Data Generator ---
class SignalDataGenerator(Sequence):
    def __init__(self, num_samples, batch_size, num_peaks, signal_length):
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.num_peaks = num_peaks
        self.signal_length = signal_length

    def __len__(self):
        return math.ceil(self.num_samples / self.batch_size)

    def __getitem__(self, idx):
        batch_x, batch_y = [], []
        for _ in range(self.batch_size):
            signal, label = generate_signal(self.num_peaks, self.signal_length)
            batch_x.append(signal)
            batch_y.append(label)
        X = np.expand_dims(np.array(batch_x), -1)
        Y = np.array(batch_y)
        return X, Y

# --- Model ---
def create_peak_detector(input_length=1024, num_peaks=5):
    model = models.Sequential([
        layers.Input(shape=(input_length, 1)),
        layers.Conv1D(32, 5, padding='same', activation='relu'),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, 5, padding='same', activation='relu'),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, 5, padding='same', activation='relu'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dense(2 * num_peaks, activation='linear')
    ])
    return model

# --- Real-time Plot Callback ---
class LivePlotCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.train_loss = []
        self.val_loss = []
        self.fig, self.ax = plt.subplots()
        plt.ion()
        plt.show()

    def on_epoch_end(self, epoch, logs=None):
        self.train_loss.append(logs["loss"])
        self.val_loss.append(logs["val_loss"])
        self.ax.clear()
        self.ax.plot(self.train_loss, label="Train Loss")
        self.ax.plot(self.val_loss, label="Val Loss")
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")
        self.ax.set_title("Training Progress")
        self.ax.legend()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

# --- Main ---
def main():
    signal_length = 1024
    num_peaks = 5
    batch_size = 32
    train_samples = 5000
    val_samples = 1000
    epochs = 10

    model = create_peak_detector(signal_length, num_peaks)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.summary()

    train_gen = SignalDataGenerator(train_samples, batch_size, num_peaks, signal_length)
    val_gen = SignalDataGenerator(val_samples, batch_size, num_peaks, signal_length)

    model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=[LivePlotCallback()])

if __name__ == "__main__":
    main()
