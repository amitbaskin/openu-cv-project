import tensorflow as tf
from tf.data.Dataset import from_generator
from myconstants import *


class TestGenerator:
    def __init__(self, test_db_path):
        self.test_db_path = test_db_path

    def __call__(self):
        with h5py.File(self.test_db_path, 'r') as hf:
            titles = hf[DATA_KEY]
            for title in titles:
                words_indices = titles[title]
                indices_lst = list(titles[title])
                indices_lst.sort(key=int)
                for index in indices_lst:
                    hf_word = words_indices[index]
                    word = hf_word.attrs[WORD_KEY]
                    label = hf_word.attrs[LABEL_KEY]
                    label_lst = CLASSES_AMOUNT * [0]
                    label_lst[label] = 1
                    chars_amount = len(word.decode())
                    labels_lsts = [label_lst for i in range(chars_amount)]
                    labels_arr = np.array(labels_lsts)
                    crops_lst = list(hf_word[CROPS_KEY])
                    crops_arr = np.array(crops_lst)
                    yield crops_arr, labels_arr


def get_test_batches(test_db_path):
    output_shapes = ([None, *IMG_SHAPE], [None, CLASSES_AMOUNT])
    return from_generator(TestGenerator(test_db_path),
                          output_types=(tf.float32, tf.float32), output_shapes=output_shapes)
