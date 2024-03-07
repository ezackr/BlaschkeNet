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
    return np.angle(f)


def main():
    x_values = np.linspace(0, 1000 * 2 * np.pi, 1000)
    f = np.cos(x_values) + 1
    sampling_rate = 1
    # plot_waveform(f, sampling_rate=sampling_rate, title="Original function")
    f_plus = remove_negative_frequencies(f)
    plot_waveform(f_plus.real, sampling_rate=sampling_rate, title="F on the real axis")
    plot_complex_plane(f_plus, title="F on the complex plane")
    f_phase = get_phase(f_plus)
    plot_waveform(f_phase, title="Phase of F")
    # l = get_absolute_logarithm(f_plus)
    # # plot_waveform(l.real, sampling_rate=sampling_rate)
    # l_plus = remove_negative_frequencies(l)
    # g_plus = np.exp(l_plus)
    # plot_waveform(g_plus.real, sampling_rate=sampling_rate, title="G on the real axis")
    # plot_complex_plane(g_plus, title="G on the complex plane")
    # g_phase = get_phase(g_plus)
    # plot_waveform(g_phase, sampling_rate=sampling_rate, title="Phase of G")
    # b_plus = f_plus / g_plus
    # plot_waveform(b_plus.real, sampling_rate=sampling_rate, title="B on the real axis")
    # plot_complex_plane(b_plus, title="B on the complex plane")
    # b_phase = get_phase(b_plus)
    # plot_waveform(b_phase.real, sampling_rate=sampling_rate, title="Phase of B")


if __name__ == "__main__":
    main()
