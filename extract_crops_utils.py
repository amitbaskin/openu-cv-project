def validate_pts(pts_bb, img, is_drop):
    height, width = img.shape[0], img.shape[1]

    for i in range(len(pts_bb)):

        if pts_bb[i][0] < 0 or pts_bb[i][0] > height - 1:
            if is_drop:
                return None
            pts_bb[i][0] = min(max(pts_bb[i][0], 0), height - 1)

        if pts_bb[i][1] < 0 or pts_bb[i][0] > width - 1:
            if is_drop:
                return None

            pts_bb[i][1] = min(max(pts_bb[i][1], 0), width - 1)

    pts_bb = pts_bb.astype(np.int32)

    h = pts_bb[2][0] - pts_bb[0][0]
    w = pts_bb[2][1] - pts_bb[0][1]

    if h < MIN_SIZE or w < MIN_SIZE:
        if is_drop:
            return None
        pts_bb = get_rect(pts_bb)
        pts_bb = pad_crop(pts_bb, img, h, w)

    return pts_bb


def get_rect(pts_bb):
    y = [p[0] for p in pts_bb]
    x = [p[1] for p in pts_bb]
    min_y, min_x = min(y), min(x)
    max_y, max_x = max(y), max(x)
    return [[min_y, min_x], [min_y, max_x], [max_y, max_x], [max_y, min_x]]


def pad_crop(pts_bb, img, h, w):
    h_pad = max((PAD_SIZE - h) // 2, h // 4)
    w_pad = max((PAD_SIZE - w) // 2, w // 4)
    height, width = img.shape[0], img.shape[1]
    pts_bb[0][0] = max(pts_bb[0][0] - h_pad, 0)
    pts_bb[1][0] = max(pts_bb[1][0] - h_pad, 0)
    pts_bb[2][0] = min(pts_bb[2][0] + h_pad, height - 1)
    pts_bb[3][0] = min(pts_bb[3][0] + h_pad, height - 1)
    pts_bb[0][1] = max(pts_bb[0][1] - w_pad, 0)
    pts_bb[1][1] = min(pts_bb[1][1] + w_pad, width - 1)
    pts_bb[2][1] = min(pts_bb[2][1] + w_pad, width - 1)
    pts_bb[3][1] = max(pts_bb[3][1] - w_pad, 0)
    return pts_bb


def get_rect_crop(img, pts_bb, is_drop=False):
    height, width = img.shape[0], img.shape[1]

    pts_bb = np.array(get_rect(pts_bb)).astype(np.int32)

    for i in range(len(pts_bb)):

        if pts_bb[i][0] < 0 or pts_bb[i][0] > height - 1:
            if is_drop:
                return None
            pts_bb[i][0] = min(max(pts_bb[i][0], 0), height - 1)

        if pts_bb[i][1] < 0 or pts_bb[i][1] > width - 1:
            if is_drop:
                return None
            pts_bb[i][1] = min(max(pts_bb[i][1], 0), width - 1)

    h = pts_bb[2][0] - pts_bb[0][0]
    w = pts_bb[2][1] - pts_bb[0][1]

    if h < MIN_SIZE or w < MIN_SIZE:
        if is_drop:
            return None

    pts_bb = pad_crop(pts_bb, img, h, w)
    crop = img[pts_bb[0][1]:pts_bb[2][1], pts_bb[0][0]:pts_bb[2][0]]
    crop = cv2.resize(crop, IMG_SIZE)
    # plt.imshow(img)
    # plt.scatter(pts_bb[0][0], pts_bb[0][1], marker="x", color="red", s=200)
    # plt.scatter(pts_bb[1][0], pts_bb[1][1], marker="x", color="blue", s=200)
    # plt.scatter(pts_bb[2][0], pts_bb[2][1], marker="x", color="green", s=200)
    # plt.scatter(pts_bb[3][0], pts_bb[3][1], marker="x", color="black", s=200)
    # plt.show()
    # plt.imshow(crop)
    # plt.show()
    return crop


def get_polygon_crop(img, pts_bb, is_drop=False):
    pts_bb = validate_pts(pts_bb, img, is_drop)

    if pts_bb is None:
        return None

    img_size = (img.shape[0], img.shape[1])
    mask = np.zeros(img_size, dtype=np.uint8)

    # pts_bb = pad_crop(pts_bb, img)
    cv2.fillPoly(mask, [pts_bb], (255))
    res = cv2.bitwise_and(img, img, mask=mask)
    rect = cv2.boundingRect(pts_bb)

    if rect[3] == 0 or rect[2] == 0:
        if is_drop:
            return None
        rect[3] += PAD_MARGIN
        rect[2] += PAD_MARGIN

    crop = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
    # crop = img[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
    crop = cv2.resize(crop, IMG_SIZE)
    # plt.imshow(crop)
    # plt.imshow(img)
    # pts_bb = pts_bb[0]
    # plt.scatter(pts_bb[0][0], pts_bb[0][1], marker="x", color="red", s=200)
    # plt.scatter(pts_bb[1][0], pts_bb[1][1], marker="x", color="blue", s=200)
    # plt.scatter(pts_bb[2][0], pts_bb[2][1], marker="x", color="green", s=200)
    # plt.scatter(pts_bb[3][0], pts_bb[3][1], marker="x", color="black", s=200)
    # plt.show()
    return crop


def write_crop(crop, title, label, item_index, path):
    try:
        title = title.decode()
    except:
        pass
    title = title.split('.')[-2]
    parent_path = os.path.join(path, str(label))
    path = os.path.join(parent_path, f'{title}_{item_index}{IMG_SUFFIX}')
    cv2.imwrite(path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    return 1


def write_all_crop(img, all_pts_bb, title, fonts_dict, fonts, path, crop_func, is_drop):
    counter = 0
    for i in range(len(all_pts_bb)):
        crop = crop_func(img, all_pts_bb[i], is_drop)
        if crop is None:
            continue
        label = get_label(fonts_dict, fonts[i])
        item_index = i
        counter += write_crop(crop, title, label, item_index, path)
    return counter


def extract_items_from_img(title, db_title, fonts_dict, path, crop_func, attr_bb, is_drop):
    items_amount = db_title.attrs[attr_bb].shape[2]
    fonts = db_title.attrs[FONT_KEY]
    img = get_img(db_title)
    all_bb = get_all_bb(db_title, attr_bb)
    all_y_x_bb = get_all_y_x_bb(all_bb)
    all_pts_bb = get_all_pts_bb(all_y_x_bb, items_amount)
    return write_all_crop(img, all_pts_bb, title, fonts_dict, fonts, path, crop_func, is_drop)


def extract_items_from_all_imgs(dataset, titles, path, crop_func, attr_bb, is_drop=False):
    fonts_dict = get_fonts_dict()
    counter = 0
    print('\n\n\nstarting - extract_items_from_all_imgs\n\n\n')
    for i in range(len(titles)):
        title = titles[i]
        db_title = get_db_title(dataset, titles[i])
        counter += extract_items_from_img(title, db_title, fonts_dict, path, crop_func, attr_bb, is_drop)
        print(counter)
    print('\n\n\nfinished - extract_items_from_all_imgs\n\n\n')