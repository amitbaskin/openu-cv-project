import h5py
from constants import *


def get_fonts_dict():
    fonts_dict = dict()
    for i in range(len(FONTS)):
        fonts_dict[FONTS[i]] = i
    return fonts_dict


def get_label(fonts_dict, font):
    try:
        return fonts_dict[font.decode()]
    except:
        return fonts_dict[font]


def get_dataset():
    return h5py.File(DATASET_PATH)


def get_titles(dataset):
    return list(dataset[DATA_KEY].keys())


def get_test_validation_titles(titles):
    random.shuffle(titles)
    n = int(SPLIT_FACTOR * len(titles))
    return titles[:n], titles[n:]


def init_hf(path):
    try:
        hf = h5py.File(path, 'w')
    except:
        os.remove(path)
        hf = h5py.File(path, 'w')
    hf.create_group(DATA_KEY)
    return hf


def load_hf(path):
    return h5py.File(path, 'r')


def put_titles_in_hf(hf, titles_key, titles):
    hf[DATA_KEY].create_dataset(titles_key, data=titles)


def put_all_titles_in_hf(hf_titles, test_titles, validation_titles):
    put_titles_in_hf(hf_titles, TEST_KEY, test_titles)
    put_titles_in_hf(hf_titles, VALIDATION_KEY, validation_titles)


def get_db_title(dataset, title):
    return dataset[DATA_KEY][title]


def get_img(db_title):
    return db_title[:]