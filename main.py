import os
import test_model


cwd = os.getcwd()
PATH_TO_TEST = os.path.join(cwd, 'test_raw.h5')
SAVE_PATH = os.path.join(cwd, 'test_processed.h5')
RESULTS_PATH = os.path.join(cwd, 'results.csv')
MODEL_PATH = os.path.join(cwd, 'model.h5')


def main():
    test_model.get_results(PATH_TO_TEST, SAVE_PATH, RESULTS_PATH, MODEL_PATH)


if __name__ == '__main__':
    main()
