import numpy as np

from src.main.util.visual import plot_waveform, plot_complex_plane


def remove_negative_frequencies(f: np.ndarray) -> np.ndarray:
    """
    Removes negative frequencies from an input signal f. A fourier transform
    is applied to f, where each negative frequency entry is set to 0. Then, an
    inverse fourier transform is applied to recover the original signal. The
    final signal is multiplied by two to preserve the original relation. That
    is, the real component of the new waveform is equal to the original
    waveform.
    :param f: a waveform
    :return: the original waveform without negative frequencies
    """
    mean = np.mean(f)
    f_demean = f - mean
    frequency_mask = (np.fft.fftfreq(len(f)) >= 0).astype(int)
    f_fft = np.fft.fft(f_demean) * frequency_mask
    f_ifft = 2 * np.fft.ifft(f_fft)
    return f_ifft + mean


def get_absolute_logarithm(f: np.ndarray):
    """
    Computes the absolute logarithm of a given function. Uses the modulus
    function if necessary for complex values.
    :param f: a function
    :return: the log of the absolute value of f
    """
    return np.log(np.absolute(f))


def get_blaschke_decomposition(f: np.ndarray):
    """
    Returns the Blaschke decomposition for a function f.
    :param f: a function
    :return: the blaschke decomposition (F_pos, B_pos, G_pos)
    """
    f_plus = remove_negative_frequencies(f)
    l = get_absolute_logarithm(f_plus)
    l_plus = remove_negative_frequencies(l)
    g_plus = np.exp(l_plus)
    b_plus = f_plus / g_plus
    return f_plus, g_plus, b_plus


def get_phase(f: np.ndarray) -> np.ndarray:
    """
    Returns the phase of a complex-valued function f.
    :param f: a function
    :return: the phase of the function
    """
    return np.angle(f)
