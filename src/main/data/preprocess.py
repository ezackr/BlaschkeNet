from os import walk
from os.path import join
from typing import List

import numpy as np
import torch
from pydub import AudioSegment
from torch.utils.data import Dataset, TensorDataset
from tqdm import tqdm

from src.main.data.constants import LANG_LABELS
from src.main.data.fourier import get_blaschke_decomposition, get_phase


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


def process_mp3s_to_ds(mp3_dir: str, labels: list[str], num_samples: int) -> TensorDataset:
    """
    Processes the downloaded mp3s into a pytorch tensor dataset.
    :param num_samples: number of samples per label
    :param mp3_dir: MUST include trailing /
    :param labels: langs
    :returns TensorDataset
    """
    vecs = []
    for label_index, label_name in enumerate(labels):
        print(f'Processing {label_name}...')
        for i in tqdm(range(num_samples)):
            audio = mp3_to_numpy(f'{mp3_dir}{label_name}/sample{i}.mp3')
            f, g, b = get_blaschke_decomposition(audio)
            phase = get_phase(b)
            vector = torch.cat((torch.tensor(label_index).view(1), torch.from_numpy(phase)))
            vecs.append(vector)
    return TensorDataset(torch.vstack(vecs))


if __name__ == '__main__':
    td = process_mp3s_to_ds('dataset/', LANG_LABELS, 150)
    torch.save(td, 'dataset.pt')

# UNUSED

# class BlaschkeDataset(Dataset):
#     """Blaschke dataset of all downloaded mp3s."""
#
#     def __init__(self, mp3_dir: str, labels: list[str], num_samples: int):
#         """
#
#         :param mp3_dir: directory of mp3s of dataset NEEDS trailing /
#         :param labels: languages
#         :param num_samples: number of samples per label (must be consistent)
#         """
#         self.mp3_dir = mp3_dir
#         self.labels = labels
#         self.num_samples = num_samples
#
#     def __len__(self):
#         return len(self.labels) * self.num_samples
#
#     def __getitem__(self, idx):
#         label = self.labels[idx // self.num_samples]
#         sample_num = idx % self.num_samples
#
#         audio = AudioSegment.from_mp3(f'{self.mp3_dir}sample{sample_num}.mp3')
#
