import numpy as np


def remove_negative_frequencies(f: np.ndarray):
    """
    Removes negative frequencies from an input signal f. To remove negative
    frequencies, a fourier transform is applied to f, which is then multiplied
    by the ReLU function max(0, x). Then, an inverse fourier transform is
    applied to recover the original signal.
    :param f: a waveform
    :return: the waveform without negative frequencies
    """
    return 2 * np.fft.ifft(np.maximum(np.fft.fft(f), 0))
