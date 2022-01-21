from extract_crops_utils import *
from hf_utils import *


def write_crop(crop, title, label, item_index, path):
    try:
        title = title.decode()
    except:
        pass
    title = title.split('.')[-2]
    parent_path = os.path.join(path, str(label))
    path = os.path.join(parent_path, f'{title}_{item_index}{IMG_SUFFIX}')
    cv2.imwrite(path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))


def write_all_crop(img, all_pts_bb, title, img_fonts, crop_attrs):
    counter = 0
    for i in range(len(all_pts_bb)):
        crop = crop_func(img, all_pts_bb[i], crop_attrs.is_drop)
        if crop is None:
            continue
        label = get_label(img_fonts[i])
        item_index = i
        write_crop(crop, title, label, item_index, crop_attrs.path)
        counter += 1
    return counter


def extract_items_from_img(title, db_title, crop_attrs):
    items_amount = db_title.attrs[attr_bb].shape[2]
    img_fonts = db_title.attrs[FONT_KEY]
    img = get_img(db_title)
    all_bb = get_all_bb(db_title, attr_bb)
    all_y_x_bb = get_all_y_x_bb(all_bb)
    all_pts_bb = get_all_pts_bb(all_y_x_bb, items_amount)
    return write_all_crop(img, all_pts_bb, title, img_fonts, crop_attrs)


def extract_items_from_all_imgs(dataset, titles, crop_attrs):
    counter = 0
    print('\n\n\nstarting - extract_items_from_all_imgs\n\n\n')
    for i in range(len(titles)):
        title = titles[i]
        db_title = get_db_title(dataset, titles[i])
        counter += extract_items_from_img(title, db_title, crop_attrs)
        print(counter)
    print('\n\n\nfinished - extract_items_from_all_imgs\n\n\n')
