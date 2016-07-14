__author__ = 'marko'

#!/usr/local/bin/python
import logging

import cv2

import numpy as np

from scipy import misc
from scipy import ndimage
from sklearn import preprocessing

"""
    TODO: Documentation of the class and methods
"""

logger = logging.getLogger(__name__)


class Preprocess(object):
    # sharpness meter
    def __compute_acutance(self, image_data):
        gray_image = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
        image_acutance = cv2.Laplacian(gray_image, cv2.CV_64F).var()
        logger.debug('acutance: %.2f', image_acutance)
        return image_acutance

    def __deNoise(self, image_data):
        image_noisy = image_data + 0.4 * image_data.std() * np.random.random(image_data.shape)
        denoised_medium = ndimage.median_filter(image_noisy, 3)

        return denoised_medium

    def __gamma_correction(self, image_data, correction=1.0):
        img = image_data/255.0
        img = cv2.pow(img, correction)
        return np.uint8(img*255)

    def __blurrer(self, image_data):
        image_blurred = ndimage.gaussian_filter(image_data, sigma=(1, 1, 0), order=0)
        #image_blurred = cv2.medianBlur(image_data, 3)
        return image_blurred

    def __sharpner(self, image_data):
        blurred_f = ndimage.gaussian_filter(image_data, 3)
        filter_blurred_f = ndimage.gaussian_filter(blurred_f, 1)
        alpha = 30
        image_sharpened = blurred_f + alpha * (blurred_f - filter_blurred_f)

        return image_sharpened

    def __clahe(self, image_data):
        img_lab = cv2.cvtColor(image_data, cv2.COLOR_RGB2LAB)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])
        img_output = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)

        return img_output

    def blur_image(self, image_data):
        _sharpness = self.__compute_acutance(image_data)
        logger.debug('acutance %.2f', _sharpness)
        # image_data = self.__deNoise(image_data)

        image_data = self.__clahe(image_data)

        if _sharpness > 100:
            image_data = self.__blurrer(image_data)
        #image_data = self.__uniform(image_data)

        image_data = self.__gamma_correction(image_data)
        return image_data

    def sharp_image(self, image_data):
        _sharpness = self.__compute_acutance(image_data)
        image_data = self.__clahe(image_data)
        logger.debug('acutance %.2f', _sharpness)
        # image_data = self.__sharpner(image_data)
        #_uniform = self.__uniform(image_data)
        #_denoise = self.__deNoise(image_data)

        return image_data
