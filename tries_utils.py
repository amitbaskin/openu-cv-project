from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tensorflow.keras.models import model_from_json
from myconstants import *


START_COORDINATES = (30, 30)
FONT_SIZE = 100
FONT_COLOR = (255, 255, 255)
MARGIN = 10


def get_char_lst(path_to_fonts, char):
    return [f'{path_to_fonts}/Alex Brush/{char}.jpg',
            f'{path_to_fonts}/Michroma/{char}.jpg',
            f'{path_to_fonts}/Open Sans/{char}.jpg',
            f'{path_to_fonts}/Raleway/{char}.jpg',
            f'{path_to_fonts}/Roboto/{char}.jpg',
            f'{path_to_fonts}/Russo One/{char}.jpg',
            f'{path_to_fonts}/Ubuntu Mono/{char}.jpg']


def get_char_img(char, path_to_fonts):
    for f in FONTS:
        img = Image.new('RGB', IMG_SIZE)
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(f'{path_to_fonts}/{f}.ttf', size=FONT_SIZE)
        draw.text(START_COORDINATES, char, FONT_COLOR, font=font)
        x1, y1, x2, y2 = img.getbbox()
        img = np.array(img)
        img = img[y1 - MARGIN: y2 + MARGIN, x1 - MARGIN: x2 + MARGIN]
        img = cv2.resize(img, IMG_SIZE)
        plt.title(f)
        plt.imshow(img)
        plt.show()
        cv2.imwrite(f'{path_to_fonts}/{char}.jpg', img)