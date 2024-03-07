import numpy as np
import matplotlib.pyplot as plt


def plot_waveform(waveform: np.ndarray, sampling_rate: int = 44100, title: str = "", is_subplot: bool = False):
    plt.plot(np.arange(len(waveform)) / sampling_rate, waveform)
    plt.xlabel("Time (s)")
    plt.title(title)
    if not is_subplot:
        plt.show()


def plot_complex_plane(complex_values: np.ndarray, title: str = "", is_subplot: bool = False):
    plt.plot(complex_values.real, complex_values.imag)
    plt.xlabel("Real")
    plt.ylabel("Imaginary")
    plt.title(title)
    if not is_subplot:
        plt.show()


def plot_waveform_summary(waveform: np.ndarray, sampling_rate: int = 44100, name: str = ""):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plot_waveform(waveform.real, sampling_rate=sampling_rate, title=f"{name} on real axis", is_subplot=True)
    plt.subplot(1, 3, 2)
    plot_complex_plane(waveform, title=f"{name} on complex plane", is_subplot=True)
    plt.subplot(1, 3, 3)
    plot_waveform(np.unwrap(np.angle(waveform)), sampling_rate=sampling_rate, title=f"Phase of {name}", is_subplot=True)
    plt.tight_layout()
    plt.show()
