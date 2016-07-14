__author__ = 'marko'
import cv2

def simple_find_green(path):
    """Simple method that finds vegetation by
    extracting all pixels with dominant green channel"""
    img = cv2.imread(path)

    height, width, _ = img.shape
    mask = img.copy()[:,:,1]
    for i in xrange(height):
        for j in xrange(width):
            mask[i, j] = (img[i, j, 1] > img[i, j, 0] and img[i, j, 1] > img[i, j, 2])
    mask[mask == 1] = 255
    return mask
