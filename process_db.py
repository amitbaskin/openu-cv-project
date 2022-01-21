import db_utils
import crops_to_dir
import crops_to_db


def process_test_validation_db(db_path, test_path, validation_path):
    db, test_titles, validation_titles = db_utils.split_dataset(db_path)
    crop_attrs = CropAttrs(attrs_bb, validation_path, crop_func, is_drop=is_drop)
    crops_to_dir.extract_items_from_all_imgs(db, validation_titles, crop_attrs)
    crops_to_db.put_all_imgs_words_in_hf(test_titles, test_path, get_rect_crop, CHAR_BB_KEY)


def process_test_validation_training_db(db_path, test_path, validation_path, training_path):
    db, test_titles, validation_titles = db_utils.split_dataset(db_path)
    crop_attrs = CropAttrs(attrs_bb, validation_path, crop_func, is_drop=is_drop)
    crops_to_dir.extract_items_from_all_imgs(db, validation_titles, crop_attrs)
    crops_to_dir.extract_items_from_all_imgs(db, validation_titles, crop_attrs)
    crops_to_db.put_all_imgs_words_in_hf(test_titles, test_path, get_rect_crop, CHAR_BB_KEY)
