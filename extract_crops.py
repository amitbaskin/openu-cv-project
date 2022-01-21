from crop_utils import *
import hf_utils
from db_attributes import *


def write_crop(img_attrs):
    title = img_attrs.get_title().split('.')[-2]
    parent_path = os.path.join(img_attrs.get_save_path(), str(label))
    save_path = os.path.join(parent_path, f'{title}_{item_index}{IMG_SUFFIX}')
    cv2.imwrite(save_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))


def write_all_crop(title_attrs, crop_attrs):
    counter = 0
    for i in range(len(all_pts_bb)):
        crop = crop_func(title_attrs.get_img(), title_attrs.get_all_pts_bb[i], crop_attrs.get_is_drop())
        if crop is None:
            continue
        label = hf_utils.get_label(title_attrs.get_fonts()[i])
        item_index = i
        img_attrs = ImgAttrs(crop, title, label, item_index, crop_attrs.get_save_path())
        write_crop(img_attrs)
        counter += 1
    return counter


def extract_items_from_all_imgs(db, titles, crop_attrs):
    counter = 0
    print('\n\n\nstarting - extract_items_from_all_imgs\n\n\n')
    for i in range(len(titles)):
        title_attrs = TitleAttrs(db, titles[i], crop_attrs.get_attrs_bb())
        counter += write_all_crop(title_attrs, crop_attrs)
        print(counter)
    print('\n\n\nfinished - extract_items_from_all_imgs\n\n\n')
