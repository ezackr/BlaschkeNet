from src.main.data.downloader import _get_number_of_samples_in_lang


def main():
    assert _get_number_of_samples_in_lang('german') == 45


if __name__ == '__main__':
    main()
