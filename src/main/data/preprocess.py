from os import walk
from os.path import join
from typing import List

import numpy as np
from pydub import AudioSegment
from tqdm import tqdm


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
