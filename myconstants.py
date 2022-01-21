CLASSES_AMOUNT = 7
IMG_SIZE = (224, 224)
IMG_SHAPE = IMG_SIZE + (3,)
FONTS = ('Alex Brush', 'Michroma', 'Raleway', 'Russo One', 'Open Sans', 'Ubuntu Mono', 'Roboto')
DATA_KEY = 'data'
CHAR_BB_KEY = 'charBB'
WORD_BB_KEY = 'wordBB'
FONT_KEY = 'font'
TEXT_KEY = 'txt'
WORD_KEY = 'word'
LABEL_KEY = 'label'
CROPS_KEY = 'crops'
TEST_KEY = 'test'
VALIDATION_KEY = 'validation'
TRAINING_KEY = 'training'
IMG_SUFFIX = '.jpg'
WEIGHTS = 'imagenet'
ACTIVATION = 'relu'
MIN_SIZE = 20
PAD_SIZE = 30
TWO_SPLIT_FACTOR = 0.5
THREE_SPLIT_FACTOR = 0.1
BATCH_SIZE = 32


class FontsDict:
    dictionary = dict()
    for i in range(len(FONTS)):
        dictionary[FONTS[i]] = i

    @staticmethod
    def get_dict():
        return dictionary
