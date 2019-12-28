import os
from glob import glob

import cv2

from cfgs import IMAGE_FOLDER

foreground_files = glob(os.path.join(IMAGE_FOLDER, 'plane', '*.jpg'))
background_files = glob(os.path.join(IMAGE_FOLDER, 'sky', '*.jpg'))


def resize(filenames):
    '''
    Resize a list of images
    :param filenames: image names(list)
    :return: None
    '''
    for file in filenames:
        img = cv2.imread(file)
        img = cv2.resize(img, dsize=(240, 240))
        cv2.imwrite(file, img)


if __name__ == '__main__':
    resize(foreground_files)
    resize(background_files)
