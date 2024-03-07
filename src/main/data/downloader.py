from bs4 import BeautifulSoup

LANG_URL = 'http://accent.gmu.edu/browse_language.php?function=find&language='
SAMPLE_URL = 'http://accent.gmu.edu/soundtracks/'


def fetch_dataset(langs: list[str], target_dir: str, sample_size=5, max_samples=100):
    """
    Downloads all satisfactory language samples from the speech-accent-archive
    and processes them to remove leading and trailing silence.

    If a sample (after processing) is longer than sample_size, it is cropped. If
    it is shorter it is thrown out. Samples are saved in mp3 form to the desired
    folder with a subdirectory for each language.

    :param target_dir:
    :param max_samples: maximum number of samples to take per language
    :param langs: list of names of language to download, all lowercase
    :param sample_size: length of desired samples in seconds
    :return: dictionary with langs as keys and number of satisfactory
             samples saved as values.
    """


def _remove_silence(file: str):
    """
    Removes leading and trailing silence from a mp3.
    :param file: path to mp3
    :return:
    """


def _get_number_of_samples_in_lang(lang: str):
    """
    Returns number of samples available on archive for desired language
    :param lang:
    :return:
    """


def _download_file(url: str, target_path: str):
    """
    Downloads file at url to target_path.
    :param url:
    :return:
    """