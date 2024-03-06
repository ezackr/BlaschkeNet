import numpy as np
import matplotlib.pyplot as plt


def plot_waveform(waveform: np.ndarray, sampling_rate: int = 44100):
    plt.plot(x=len(waveform) / sampling_rate, y=waveform)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()
