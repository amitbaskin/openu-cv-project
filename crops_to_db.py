from myconstants import *
from crop_utils import *
from db_utils import *


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
    label = word_attrs.get_label()
    if label is not None:
        hf[DATA_KEY][title][word_key].attrs[LABEL_KEY] = label
    hf[DATA_KEY][title][word_key].attrs[WORD_KEY] = word
    return len(word)


def put_all_img_words_in_hf(title_attrs, crop_attrs):
    hf[DATA_KEY].create_group(title)
    char_offset = 0
    for word_index in range(len(words)):
        label = None
        if title_attrs.get_is_labeled():
            label = get_label(title_attrs.get_fonts()[offset])
        word_attrs = WordAttrs(title_attrs, crop_attrs, word_index, char_offset, label)
        char_offset += put_word_in_hf(word_attrs)
    return char_offset


def put_all_imgs_words_in_hf(db_path, titles, crop_attrs, is_labeled):
    hf = init_db(crop_attrs.get_save_path())
    db = get_db(db_path)
    counter = 0
    print('\n\n\nstarting - put_all_imgs_words_in_hf\n\n\n')
    for i in range(len(titles)):
        title_attrs = TitleAttrs(hf, db, titles[i], is_labeled)
        counter += put_all_img_words_in_hf(title_attrs, crop_attrs)
        print(counter)
    hf.close()
    print('\n\n\nfinished - put_all_imgs_words_in_hf\n\n\n')


def process_test_db(path_to_test, save_path):
    crop_attrs = CropAttrs(CHAR_BB_KEY, save_path, crop_utils.get_rect_crop, is_drop=False)
    titles = get_titles_from_db_path(path_to_test)
    put_all_imgs_words_in_hf(path_to_test, titles, crop_attrs, is_labeled=False)
