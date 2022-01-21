import matplotlib.pyplot as plt
from constants import *


def get_rect(pts_bb):
    y = [p[0] for p in pts_bb]
    x = [p[1] for p in pts_bb]
    min_y, min_x = min(y), min(x)
    max_y, max_x = max(y), max(x)
    return [[min_y, min_x], [min_y, max_x], [max_y, max_x], [max_y, min_x]]


def validate_pts_bb(pts_bb, is_drop):
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

    return pts_bb


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


def get_rect_crop(img, pts_bb, is_drop):
    pts_bb = validate_pts_bb(pts_bb, is_drop)
    if pts_bb is None:
        return None
    pts_bb = pad_crop(pts_bb, img, h, w)
    crop = img[pts_bb[0][1]:pts_bb[2][1], pts_bb[0][0]:pts_bb[2][0]]
    crop = cv2.resize(crop, IMG_SIZE)
    return crop


def get_polygon_crop(img, pts_bb, is_drop=False):
    pts_bb = validate_pts(pts_bb, img, is_drop)
    if pts_bb is None:
        return None
    img_size = (img.shape[0], img.shape[1])
    mask = np.zeros(img_size, dtype=np.uint8)
    cv2.fillPoly(mask, [pts_bb], 255)
    res = cv2.bitwise_and(img, img, mask=mask)
    rect = cv2.boundingRect(pts_bb)
    if rect[3] == 0 or rect[2] == 0:
        if is_drop:
            return None
        rect[3] += PAD_MARGIN
        rect[2] += PAD_MARGIN
    crop = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
    return cv2.resize(crop, IMG_SIZE)


def plot_crop(img, pts_bb, crop):
    plt.imshow(img)
    plt.scatter(pts_bb[0][0], pts_bb[0][1], marker="x", color="red", s=200)
    plt.scatter(pts_bb[1][0], pts_bb[1][1], marker="x", color="blue", s=200)
    plt.scatter(pts_bb[2][0], pts_bb[2][1], marker="x", color="green", s=200)
    plt.scatter(pts_bb[3][0], pts_bb[3][1], marker="x", color="black", s=200)
    plt.show()
    plt.imshow(crop)
    plt.show()
