import tensorflow as tf
from tf.keras.preprocessing import image_dataset_from_directory
from tf.data.Dataset import sample_from_datasets
from tf.data import AUTOTUNE
from myconstants import *


def get_set_from_dir(dir_path):
    return image_dataset_from_directory(dir_path, labels='inferred', label_mode='int',
                                        shuffle=True,
                                        batch_size=BATCH_SIZE,
                                        image_size=IMG_SIZE)


def get_training(paths_lst):
    datasets = [get_set_from_dir(p) for p in paths_lst]
    datasets = [ds.map(lambda x, y: (aug(x) / 255.0, tf.one_hot(y, 7))) for ds in datasets]
    training = sample_from_datasets(datasets, weights=[1 / len(paths_lst) for _ in range(len(paths_lst))])
    return training.prefetch(tf.data.AUTOTUNE)


def get_validation(validation_path):
    validation = get_set_from_dir(validation_path)
    validation = validation.map(lambda x, y: (x / 255.0, tf.one_hot(y, 7)))
    return validation.prefetch(tf.data.AUTOTUNE)
