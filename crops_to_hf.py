from myconstants import *
from extract_crops_utils import *
from hf_utils import *


class TitleAttrs:
    def __init__(self, hf, db, title):
        self.hf = hf
        self.db = db
        self.title = title

    def get_hf(self):
        return self.hf

    def get_db(self):
        return self.db

    def get_title(self):
        return self.title

    def get_db_title(self):
        return dataset[DATA_KEY][self.get_title()]

    def get_words(self):
        return self.get_db_title().attrs[TEXT_KEY]

    def get_img(self):
        return self.get_db_title()[:]

    def get_single_word(self, word_index):
        return self.get_word(word_index)

    def get_fonts(self):
        return self.get_db_title().attrs[FONT_KEY]


class WordAttrs:
    def __init__(self, title_attrs, crop_attrs, word_index, char_offset, label):
        self.title_attrs = title_attrs
        self.crop_attrs = crop_attrs
        self.word_index = word_index
        self.char_offset = char_offset
        self.label = label

    def get_title_attrs(self):
        return self.title_attrs

    def get_crop_attrs(self):
        return self.crop_attrs

    def get_word_index(self):
        return self.word_index

    def get_char_offset(self):
        return self.char_offset

    def get_label(self):
        return self.label

    def get_word(self):
        self.title_attrs.get_single_word(self.word_index)

    def get_db_title(self):
        return self.get_title_attrs().get_db_title()

    def get_attrs_bb(self):
        return self.get_title_attrs().get_attrs_bb()

    def get_crop_func(self):
        return self.get_crop_attrs().get_crop_func()

    def get_img(self):
        return self.get_title_attrs().get_img()

    def get_hf(self):
        return self.get_title_attrs().get_hf()


def put_word_in_hf(word_attrs):
    word = word_attrs.get_word()
    amount = len(word)
    crops = np.empty((amount, *IMG_SHAPE))
    for i in range(amount):
        pts_bb = get_pts_bb(word_attrs.get_db_title(), word_attrs.get_char_offset() + i, word_attrs.get_attrs_bb())
        crops[i] = word_attrs.get_crop_func()(word_attrs.get_img(), pts_bb, False)
    word_key = str(word_index)
    hf = word_attrs.get_hf()
    hf[DATA_KEY][title].create_group(word_key)
    hf[DATA_KEY][title][word_key].create_dataset(CROPS_KEY, data=crops)
    hf[DATA_KEY][title][word_key].attrs[LABEL_KEY] = word_attrs.get_label()
    hf[DATA_KEY][title][word_key].attrs[WORD_KEY] = word
    return len(word)


def put_all_img_words_in_hf(title_attrs, crop_attrs):
    hf[DATA_KEY].create_group(title)
    char_offset = 0
    for word_index in range(len(words)):
        label = get_label(title_attrs.get_fonts()[offset])
        word_attrs = WordAttrs(title_attrs, crop_attrs, word_index, char_offset, label)
        char_offset += put_word_in_hf(word_attrs)
    return char_offset


def put_all_imgs_words_in_hf(titles, crop_attrs):
    hf = init_hf(path)
    dataset = get_dataset()
    counter = 0
    print('\n\n\nstarting - put_all_imgs_words_in_hf\n\n\n')
    for i in range(len(titles)):
        title_attrs = TitleAttrs(hf, dataset, titles[i])
        counter += put_all_img_words_in_hf(title_attrs, crop_attrs)
        print(counter)
    hf.close()
    print('\n\n\nfinished - put_all_imgs_words_in_hf\n\n\n')
