from typing import List

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


def get_fourier_coefficients(
        f: np.ndarray,
        degree: int = None,
        sampling_rate: int = 44100
) -> List[float]:
    """
    Computes the fourier coefficients of a given waveform function f.
    :param f: a waveform
    :param degree: the number of terms in the fourier series
    :param sampling_rate: the sampling rate of the waveform
    :return: a list of fourier coefficients
    """
    # use maximum degree polynomial if none provided
    if not degree:
        degree = len(f)
    time_values = np.arange(0, len(f)) / sampling_rate
    coefficients = []
    for i in range(degree + 1):
        c_i = np.sum(f * np.exp(-1j * 2 * np.pi * i * time_values / len(f))) / len(f)
        coefficients.append(c_i)
    return coefficients


def get_absolute_logarithm(f: np.ndarray):
    """
    Computes the absolute logarithm of a given function. Uses the modulus
    function if necessary for complex values.
    :param f: a function
    :return: the log of the absolute value of f
    """
    return np.log(np.absolute(f))
