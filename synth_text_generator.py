import os

def generate_data(use_preproc_bg_py_path):
    to_run = f'''
    pip2.7 install pygame
    pip2.7 install wget
    python2.7 {use_preproc_bg_py_path}
    '''
    os.system(to_run)
