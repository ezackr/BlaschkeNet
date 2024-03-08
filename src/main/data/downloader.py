from bs4 import BeautifulSoup
from urllib.request import urlopen, urlretrieve
from pydub.silence import detect_leading_silence
from pydub import AudioSegment, silence
import os
from pathlib import Path
from tqdm import tqdm as tqdm

from src.main.data.constants import LANG_LABELS

LANG_URL = 'http://accent.gmu.edu/browse_language.php?function=find&language='
SAMPLE_URL = 'http://accent.gmu.edu/soundtracks/'


def fetch_dataset(langs: list[str], target_dir: str, sample_duration=5, max_samples=100):
    """
    Downloads all satisfactory language samples from the speech-accent-archive
    and processes them to remove leading and trailing silence.

    If a sample (after processing) is longer than sample_size, it is cropped. If
    it is shorter it is thrown out. Samples are saved in mp3 form to the desired
    folder with a subdirectory for each language.

    :param target_dir: must include trailing /
    :param max_samples: maximum number of samples to take per language
    :param langs: list of names of language to download, all lowercase
    :param sample_duration: length of desired samples in seconds
    :return: dictionary with langs as keys and number of satisfactory
             samples saved as values.
    """
    dic = {}
    for lang in langs:
        num_samples = min(_get_number_of_samples_in_lang(lang), max_samples)
        sample_index = 0

        Path(f'{target_dir}{lang}').mkdir(parents=True, exist_ok=True)
        print(f'Downloading samples for {lang}...')

        for i in tqdm(range(num_samples)):
            download_index = i+1
            sample_path = f'{target_dir}{lang}/sample{sample_index}.mp3'

            urlretrieve(f'{SAMPLE_URL}{lang}{download_index}.mp3', sample_path)

            audio = AudioSegment.from_mp3(sample_path)
            audio = _remove_silence(audio)

            if audio.duration_seconds < sample_duration:
                # Non-satisfactory sample, throw it out
                os.remove(sample_path)
            else:
                audio = audio.set_frame_rate(44100)
                audio = audio[:sample_duration*1000]
                audio.export(sample_path, bitrate='128k')
                sample_index += 1

        dic[lang] = sample_index

    return dic


def _remove_silence(audio: AudioSegment):
    """
    Removes leading and trailing silence from an AudioSegment
    :param audio:
    :return: cropped AudioSegment
    """

    cropped = audio[detect_leading_silence(audio):]
    cropped = cropped.reverse()[detect_leading_silence(cropped.reverse()):].reverse()

    return cropped


def _get_number_of_samples_in_lang(lang: str):
    """
    Returns number of samples available on archive for desired language
    :param lang:
    :return:
    """
    soup = BeautifulSoup(urlopen(LANG_URL + lang), 'html.parser')
    return len(soup.find_all('p'))-1


# Run this file from cmdline in the directory you want to put dataset/ in :)
if __name__ == '__main__':
    fetch_dataset(LANG_LABELS, 'dataset/', max_samples=10)

