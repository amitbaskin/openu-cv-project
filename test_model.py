import csv
import numpy as np
import h5py
import tensorflow as tf
from tf.keras.models import load_model
from test_generator import get_test_batches
from myconstants import *
from db_attributes import *
import crop_utils


def predict(model, X):
    preds = model.predict(X)
    d = dict()
    for i in range(7):
        d[i] = 0
    for p in preds:
        for i in range(7):
            d[i] += p[i]
    mx = max(d, key=d.get)
    return mx


def evaluate_batch(pred, Y):
    if Y[0][pred] == 1:
        return Y.shape[0]
    else:
        return 0


def evaluate_helper(model, test_db_path):
    amount_of_samples = 0
    hits = 0
    batches = get_test_batches(test_db_path)
    for batch in batches:
        amount_of_samples += batch[0].shape[0]
        print(amount_of_samples)
        pred = predict(model, batch[0])
        hits += evaluate_batch(pred, batch[1])
    return hits / amount_of_samples


def evaluate(test_db_path, save_path, path_to_model):
    process_test_db(path_to_test, save_path)
    model = load_model(path_to_model)
    print(evaluate_helper(model, test_db_path))


def get_results_helper(model, db_path, results_path):
    examples_amount = 0
    with open(results_path, 'w') as results:
        writer = csv.writer(results)
        header = ['image', 'char', *FONTS]
        writer.writerow(header)
        with h5py.File(db_path, 'r') as hf:
            titles = hf[DATA_KEY]
            for title in titles:
                words_indices = titles[title]
                indices_lst = list(titles[title])
                indices_lst.sort(key=int)
                for index in indices_lst:
                    hf_word = words_indices[index]
                    word = hf_word.attrs[WORD_KEY].decode()
                    chars_amount = len(word)
                    crops_lst = list(hf_word[CROPS_KEY])
                    crops_arr = np.array(crops_lst)
                    prediction = predict(model, crops_arr)
                    examples_amount += chars_amount
                    print(examples_amount)
                    for i in range(chars_amount):
                        label_lst = CLASSES_AMOUNT * [0]
                        label_lst[prediction] = 1
                        row = [title, word[i]] + label_lst
                        writer.writerow(row)


def get_results(path_to_test, save_path, results_path, path_to_model):
    process_test_db(path_to_test, save_path)
    model = load_model(path_to_model)
    export_results(model, save_path, results_path)
