
from src.main.data.downloader import _get_number_of_samples_in_lang, fetch_dataset


def main():
    assert _get_number_of_samples_in_lang('german') == 45
    assert _get_number_of_samples_in_lang('bosnian') == 12
    assert _get_number_of_samples_in_lang('english') == 660
    fetch_dataset(['english', 'german', 'bengali'], 'download_test/', max_samples=5)


if __name__ == '__main__':
    main()

