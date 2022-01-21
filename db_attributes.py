import bb_utils


class CropAttrs:
    def __init__(self, attrs_bb, save_path, crop_func, is_drop):
        self.attrs_bb = attrs_bb
        self.save_path = save_path
        self.crop_func = crop_func
        self.is_drop = is_drop

    def get_attrs_bb(self):
        return self.attrs_bb

    def get_save_path(self):
        return self.save_path

    def get_crop_func(self):
        return self.crop_func

    def get_is_drop(self):
        return self.is_drop

class ImgAttrs:
    def __init__(self, img, title, label, item_index, save_path):
        self.img = img
        self.label = label
        self.item_index = item_index
        self.save_path = save_path
        try:
            self.title = title.decode()
        except:
            self.title = title

    def get_img(self):
        return self.img

    def get_title(self):
        return self.title

    def get_label(self):
        return self.label

    def get_item_index(self):
        return self.item_index

    def get_save_path(self):
        return self.save_path


class TitleAttrs:
    def __init__(self, db, title, attrs_bb, is_labeled=True, hf=None):
        self.hf = hf
        self.db = db
        self.title = title
        self.attrs_bb = attrs_bb
        self.is_labeled = is_labeled

    def get_hf(self):
        return self.hf

    def get_db(self):
        return self.db

    def get_title(self):
        return self.title

    def get_attrs_bb(self):
        return self.attrs_bb

    def get_is_labeled(self):
        return self.is_labeled

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

    def get_items_amount(self):
        return self.get_db_title().attrs[self.get_attrs_bb()].shape[2]

    def get_all_pts_bb(self):
        all_bb = bb_utils.get_all_bb(self.get_db_title(), self.get_attrs_bb())
        all_y_x_bb = bb_utils.get_all_y_x_bb(all_bb)
        return bb_utils.get_all_pts_bb(all_y_x_bb, self.get_items_amount())


class WordAttrs:
    def __init__(self, title_attrs, crop_attrs, word_index, label, char_offset=None):
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

