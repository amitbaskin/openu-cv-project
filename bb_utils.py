import h5py
from constants import *
from db_utils import *


def get_all_bb(db_title, attr_bb):
    return db_title.attrs[attr_bb]


def get_all_y_x_bb(all_bb):
    return all_bb[0], all_bb[1]


def get_all_pts_bb(all_y_x_bb, items_amount):
    all_pts_bb = []
    for item_index in range(items_amount):
        y_bb, x_bb = get_y_x_bb(*all_y_x_bb, item_index)
        all_pts_bb.append(get_pts_bb_helper(y_bb, x_bb))
    return all_pts_bb
