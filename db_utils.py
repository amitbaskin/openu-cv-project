import h5py
from myconstants import *


def get_label(font):
    try:
        return fonts_dict[font.decode()]
    except:
        return fonts_dict[font]


def get_db(db_path):
    return h5py.File(db_path)


def get_titles_from_db(dataset):
    return list(dataset[DATA_KEY].keys())


def get_titles_from_db_path(db_path):
    dataset = get_db(db_path)
    return get_titles_from_db(dataset)


def get_test_validation_titles(titles):
    random.shuffle(titles)
    n = int(TWO_SPLIT_FACTOR * len(titles))
    return titles[:n], titles[n:]


def get_test_validation_training_titles(titles):
    random.shuffle(titles)
    n = int(THREE_SPLIT_FACTOR * len(titles))
    return titles[:n], titles[n:2*n], titles[2*n:]


def init_db(path):
    try:
        hf = h5py.File(path, 'w')
    except:
        os.remove(path)
        hf = h5py.File(path, 'w')
    hf.create_group(DATA_KEY)
    return hf


def load_db(path):
    return h5py.File(path, 'r')


def put_titles_in_db(db, titles_key, titles):
    db[DATA_KEY].create_dataset(titles_key, data=titles)


def put_test_validation_in_db(hf_titles, test_titles, validation_titles):
    put_titles_in_db(hf_titles, TEST_KEY, test_titles)
    put_titles_in_db(hf_titles, VALIDATION_KEY, validation_titles)


def put_test_validation_training_in_db(hf_titles, test_titles, validation_titles, training_titles):
    put_titles_in_db(hf_titles, TEST_KEY, test_titles)
    put_titles_in_db(hf_titles, VALIDATION_KEY, validation_titles)
    put_titles_in_db(hf_titles, TRAINING_KEY, training_titles)


def split_test_validation(input_db_path):
    input_db = get_db(input_db_path)
    titles = get_titles_from_db(input_db)
    test_titles, validation_titles = get_test_validation_titles(titles)
    output_db = init_db(titles_path)
    put_test_validation_in_db(output_db, test_titles, validation_titles)
    db.close()
    return input_db, test_titles, validation_titles


def split_test_validation_training(input_db_path):
    input_db = get_db(input_db_path)
    titles = get_titles_from_db(input_db)
    test_titles, validation_titles, training_titles = get_test_validation_training_titles(titles)
    output_db = init_db(titles_path)
    put_test_validation_training_in_db(output_db, test_titles, validation_titles, training_titles)
    db.close()
    return input_db, test_titles, validation_titles, training_titles
