import numpy as np


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
    f_mean = np.mean(f)
    f_demean = f - f_mean
    f_fft = np.fft.fft(f_demean)
    f_fft_shift = np.fft.fftshift(f_fft)
    f_fft_shift[len(f_fft_shift) // 2:] = 0
    f_ifft_shift = np.fft.ifftshift(f_fft_shift)
    f_ifft = 2 * np.fft.ifft(f_ifft_shift)
    return f_ifft + f_mean


def get_fourier_coefficients(f: np.ndarray):
    """
    Computes the fourier coefficients of a given waveform function f.
    :param f: a function
    :return: the fourier coefficients of f
    """
    return np.fft.fft(f)


def get_absolute_logarithm(f: np.ndarray):
    """
    Computes the absolute logarithm of a given function. Uses the modulus
    function if necessary for complex values.
    :param f: a function
    :return: the log of the absolute value of f
    """
    return np.log(np.absolute(f))


def get_positive_fourier_estimate(f: np.ndarray):
    """
    Given an arbitrary function f, recovers a positive fourier estimate F_pos.
    :param f: a function
    :return: the positive fourier estimate
    """
    f_pos = remove_negative_frequencies(f)
    f_pos_coeff = get_fourier_coefficients(f_pos)
    return np.fft.ifft(f_pos_coeff)


def get_blaschke_decomposition(f: np.ndarray):
    """
    Returns the Blaschke decomposition for a function f.
    :param f: a function
    :return: the blaschke decomposition (F_pos, B_pos, G_pos)
    """
    F_pos = get_positive_fourier_estimate(f)
    l = get_absolute_logarithm(F_pos)
    L_pos = get_positive_fourier_estimate(l)
    G_pos = np.exp(L_pos)
    B_pos = F_pos / G_pos
    return F_pos, G_pos, B_pos


def get_phase(f: np.ndarray) -> np.ndarray:
    """
    Returns the phase of a complex-valued function f.
    :param f: a function
    :return: the phase of the function
    """
    return -1j * np.log(f / np.absolute(f))
