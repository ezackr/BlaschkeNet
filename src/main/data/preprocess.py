import math
from os import walk
from os.path import join
from typing import List

import numpy as np
from pydub import AudioSegment
from tqdm import tqdm

from src.main.util import root_dir


def mp3_to_numpy(path: str) -> np.ndarray:
    """
    Converts an MP3 file to its corresponding numpy waveform.
    :param path:
    :return:
    """
    audio = AudioSegment.from_mp3(path)
    audio_array = np.array(audio.get_array_of_samples())
    return audio_array


def mp3_to_numpy_dir(mp3_dir: str) -> List[np.ndarray]:
    """
    Preprocesses a directory of MP3 files into a list of numpy waveforms.
    :param mp3_dir: a directory of MP3 files
    :return: the corresponding list of numpy waveforms
    """
    waveforms = []
    for root, _, files in walk(mp3_dir):
        for file in tqdm(files):
            abs_path = join(root, file)
            waveform = mp3_to_numpy(abs_path)
            waveforms.append(waveform)
    return waveforms


def _nearest_power_of_two(n: int) -> int:
    """
    Finds the smallest power of two greater than or equal to n.
    :param n: an integer
    :return: the smallest power of two greater than or equal to n
    """
    exp = int(math.log2(n))
    if 2 ** exp == n:
        # n is a power of two
        return n
    else:
        return 2 ** (exp + 1)


def pad_waveforms(waveforms: List[np.ndarray]) -> np.ndarray:
    """
    Converts a list of variable-length waveforms into a single array by
    padding waveforms with zeros.
    :param waveforms: the original waveforms
    :return: all waveforms as a single array with padded zeros
    """
    max_waveform_length = max([len(waveform) for waveform in waveforms])
    max_pad_length = _nearest_power_of_two(max_waveform_length)
    padded_waveforms = np.zeros(shape=(len(waveforms), max_pad_length))
    for i, waveform in enumerate(waveforms):
        padded_waveforms[i, :len(waveform)] = waveform
    return padded_waveforms


def main():
    mp3_dir = join(root_dir, "dataset", "archive", "recordings")
    waveforms = mp3_to_numpy_dir(mp3_dir)
    padded_waveforms = pad_waveforms(waveforms)


if __name__ == "__main__":
    main()
