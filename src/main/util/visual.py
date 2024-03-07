import numpy as np
import matplotlib.pyplot as plt


def plot_waveform(waveform: np.ndarray, sampling_rate: int = 44100, title: str = ""):
    plt.plot(np.arange(len(waveform)) / sampling_rate, waveform)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.show()


def plot_complex_plane(complex_values: np.ndarray, title: str = ""):
    plt.plot(complex_values.real, complex_values.imag)
    plt.xlabel("Real")
    plt.ylabel("Imaginary")
    plt.title(title)
    plt.show()
