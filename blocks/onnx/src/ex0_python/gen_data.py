#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os

def generate_signal(num_peaks, signal_length=1024, noise_level=0.05, asymmetry_factor=0.2):
    """
    Generate a synthetic 1D power spectrum with multiple peaks.
    Peaks are roughly Gaussian with some asymmetry.
    """
    x = np.linspace(0, signal_length, signal_length)
    signal = np.zeros_like(x)
    peak_positions = np.random.randint(50, signal_length - 50, num_peaks)
    fwhm_values = np.random.uniform(10, 50, num_peaks)
    amplitudes = np.random.uniform(0.5, 2.0, num_peaks)

    for pos, fwhm, amp in zip(peak_positions, fwhm_values, amplitudes):
        stddev = fwhm / (2 * np.sqrt(2 * np.log(2)))
        # Add asymmetry
        asymmetry = np.random.uniform(-asymmetry_factor, asymmetry_factor)
        gaussian = amp * np.exp(-((x - pos) ** 2) / (2 * stddev ** 2))
        # Apply linear asymmetry: peak height slightly changes depending on position
        asymmetric_gaussian = gaussian * (1 + asymmetry * (x - pos) / fwhm)
        asymmetric_gaussian = np.clip(asymmetric_gaussian, 0, None)
        signal += asymmetric_gaussian

    # Add Gaussian noise
    noise = np.random.normal(0, noise_level * np.max(signal), signal_length)
    signal += noise
    return x, signal, peak_positions, fwhm_values

def create_dataset(num_samples=100000, signal_length=1024, num_peaks=10):
    """
    Create a dataset of synthetic signals and their ground-truth peak parameters.
    The output is a tuple (X, Y) where:
    X: [num_samples, signal_length] array of signals
    Y: [num_samples, 2*num_peaks] array containing peak_positions followed by fwhms
    """
    inputs = []
    outputs = []
    for _ in range(num_samples):
        _, signal, positions, fwhms = generate_signal(num_peaks=num_peaks, signal_length=signal_length)
        # Sort by position to have a consistent order (not strictly necessary, but cleaner)
        sort_idx = np.argsort(positions)
        positions = positions[sort_idx]
        fwhms = fwhms[sort_idx]

        inputs.append(signal)
        outputs.append(np.hstack([positions, fwhms]))
    return np.array(inputs), np.array(outputs)

if __name__ == "__main__":
    # Generate and plot an example signal
    x, signal, positions, fwhms = generate_signal(num_peaks=10, signal_length=1024)
    plt.plot(x, signal, label="Signal")
    plt.scatter(positions, signal[positions], color="red", label="Detected Peaks")
    plt.title("Example Synthetic Power Spectrum")
    plt.xlabel("Frequency Bin")
    plt.ylabel("Power")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Create a training set and validation set, and save to disk
    X_train, y_train = create_dataset(num_samples=800000, num_peaks=10, signal_length=1024)
    X_val, y_val = create_dataset(num_samples=2000000, num_peaks=10, signal_length=1024)

    if not os.path.exists('data'):
        os.makedirs('data')

    np.savez('data/training_data.npz', X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
    print("Training data saved to data/training_data.npz")


