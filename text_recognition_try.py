# !pip install keras-ocr
# !pip install git+https://github.com/faustomorales/keras-ocr.git#egg=keras-ocr
import matplotlib.pyplot as plt
import keras_ocr
import cv2
import numpy as np


def extract_words(img_path):
    img = cv2.imread(img_path)
    images = [img]
    # Each list of predictions in prediction_groups is a list of
    # (word, box) tuples.
    prediction_groups = pipeline.recognize(images)
    predictions = prediction_groups[0]
    for word, box in predictions:
        box = box.astype(np.uint8)
        print(word)
        print(box)
        h, w, _ = img.shape
        y = [b[1] for b in box]
        x = [b[0] for b in box]
        y0 = min(y)
        y1 = max(y)
        x0 = min(x)
        x1 = max(x)
        y0 = max(y0 - 1, 0)
        y1 = min(y1 + 1, h)
        x0 = max(x0 - 1, 0)
        x1 = min(x1 + 1, w)
        plt.title(word)
        plt.imshow(img[y0:y1, x0:x1])
        plt.show()
