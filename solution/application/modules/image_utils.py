__author__ = 'marko'

import cv2
import matplotlib.pyplot as plt
import numpy as np

def plt_imshow(img):
    """ Util method used to display opencv images using matplotlib"""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)

def cv2_imshow(title, img):
    """
    Util method used to display images in new titled windows
    :param title:
    :param img:
    :return:
    """
    cv2.imshow(title, img)
    if cv2.waitKey(0) >= 30:
        pass

def cv2_resize(image, x=None, y=None):
    """
    Util method used for resizing given image to fixed x or y cordinate while
     maintaining aspect ratio
    :param image:
    :param x:
    :param y:
    :return:
    """
    if not x and not y:
        raise Exception('No dimension passes')

    if x:
        # Resizing the image by his width
        r = float(x) / image.shape[1]
        dim = (x, int(image.shape[0] * r))

    elif y:
        r = float(y) / image.shape[0]
        dim = (int(image.shape[1] * r), y)

    return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

def apply_clahe_rgb(img):
    """
    Apply CLAHE on RGB images by normalazing light in LAB image representation
    :param img:
    :return:
    """
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_lab[:,:,0] = clahe.apply(img_lab[:,:,0])
    normalized = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
    return normalized

def binarized_to_rgb(img):
    """
    Returns binarized image in form (h*w*3) matrix
    :param img:
    :return:
    """
    return np.stack([img, img, img], axis=2)

