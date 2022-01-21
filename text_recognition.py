import os
import subprocess

def run_command(to_run):
    subprocess.check_output(to_run, shell=True)


def install_requirments():
    to_run = '''    
    pip install pillow==7.0.0
    pip install pytesseract==0.3.2
    pip install imutils==0.5.3
    pip install opencv-python==4.2.0.32
    '''
    run_command(to_run)


def clone_text_recognition(path):
    to_run = f'git clone https://github.com/efviodo/opencv-text-recognition {path}'
    run_command(to_run)


def run_text_recognition(text_recognition_py_path, path_to_model, img_path):
    to_run = f'python {text_recognition_py_path} --east {path_to_model} --image {img_path} --padding 0.05'
    run_command(to_run)
