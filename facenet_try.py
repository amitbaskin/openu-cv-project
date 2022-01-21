# https://www.coursera.org/learn/convolutional-neural-networks/home/welcome
import cv2
from tensorflow.keras.models import model_from_json
from tries_utils import *


def load_model(facenet_json_path, facenet_h5_path):
    json_file = open(facenet_json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(facenet_h5_path)
    return model


def img_to_encoding(img, model):
    x_train = np.expand_dims(img, axis=0)
    embedding = model.predict_on_batch(x_train)
    return embedding / np.linalg.norm(embedding, ord=2)


def verify_font(target, suspect, model):
    target_enc = img_to_encoding(target, model)
    suspect_enc = img_to_encoding(suspect, model)
    dist = np.linalg.norm(target_enc - suspect_enc)
    return dist


def predict(target, char, path_to_fonts):
    get_char_img(char, path_to_fonts)
    paths = get_char_lst(char, path_to_fonts)
    imgs = [cv2.imread(p) for p in paths]
    dists = []
    for i in range(7):
        dists.append(verify(target, imgs[i], model))
    return min(range(7), key=lambda index: dists[index])
