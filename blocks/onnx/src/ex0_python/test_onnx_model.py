#!/usr/bin/env python3
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt

def generate_signal(num_peaks=5, signal_length=1024, noise_level=0.05, asymmetry_factor=0.2):
    x = np.linspace(0, signal_length, signal_length)
    signal = np.zeros_like(x)
    peak_positions = np.random.randint(50, signal_length - 50, num_peaks)
    fwhm_values = np.random.uniform(10, 50, num_peaks)
    amplitudes = np.random.uniform(0.5, 2.0, num_peaks)

    for pos, fwhm, amp in zip(peak_positions, fwhm_values, amplitudes):
        stddev = fwhm / (2 * np.sqrt(2 * np.log(2)))
        asymmetry = np.random.uniform(-asymmetry_factor, asymmetry_factor)
        gaussian = amp * np.exp(-((x - pos) ** 2) / (2 * stddev ** 2))
        asymmetric_gaussian = gaussian * (1 + asymmetry * (x - pos) / fwhm)
        asymmetric_gaussian = np.clip(asymmetric_gaussian, 0, None)
        signal += asymmetric_gaussian

    noise = np.random.normal(0, noise_level * np.max(signal), signal_length)
    signal += noise
    # Sort peaks for easier comparison
    sort_idx = np.argsort(peak_positions)
    return x, signal, peak_positions[sort_idx], fwhm_values[sort_idx]

# Load ONNX model
session = ort.InferenceSession("peak_detector.onnx", providers=['CPUExecutionProvider'])

num_examples = 3
num_peaks = 5
signal_length = 1024
np.random.seed(42) # to make examples reproducible

fig, axes = plt.subplots(num_examples, 1, figsize=(8, 10))

for i in range(num_examples):
    x, signal, true_positions, true_fwhms = generate_signal(num_peaks=num_peaks, signal_length=signal_length)
    inp = signal[np.newaxis, :, np.newaxis].astype(np.float32)  # [1, 1024, 1]
    outputs = session.run(None, {session.get_inputs()[0].name: inp})
    pred = outputs[0][0]  # [2*num_peaks]
    pred_positions = pred[:num_peaks]
    pred_fwhms = pred[num_peaks:]

    # Sort predicted by position for easier comparison
    sort_idx = np.argsort(pred_positions)
    pred_positions = pred_positions[sort_idx]
    pred_fwhms = pred_fwhms[sort_idx]

    # Plot
    ax = axes[i]
    ax.plot(x, signal, label='Signal', zorder=1)
    ax.scatter(true_positions, signal[true_positions.astype(int)], color='red', marker='x', label='True Peaks', zorder=3)
    ax.scatter(pred_positions, signal[pred_positions.astype(int)], color='green', marker='o', facecolors='none', label='Predicted Peaks', zorder=3)
    ax.set_title(f"Example {i+1}")
    ax.legend()

plt.tight_layout()
plt.show()

