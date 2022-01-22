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


def triplet_loss(y_pred, alpha=0.2):
    """
    Implementation of the triplet loss

    Arguments:
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)

    Returns:
    loss -- real number, value of the loss
    """
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    # Step 1: Compute the (encoding) distance between the anchor and the positive
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    # Step 2: Compute the (encoding) distance between the anchor and the negative
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = pos_dist - neg_dist + alpha
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(pos_dist - neg_dist + alpha, basic_loss), axis=None)
    return loss
