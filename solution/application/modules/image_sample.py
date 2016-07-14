__author__ = 'marko'

import numpy as np
from random import randint
from skimage.feature import hessian_matrix
from skimage.morphology import disk
from skimage.filters.rank import entropy
from preprocess import Preprocess
import cv2

class ImageSample(object):
    '''Image wrapper class that is used for samples extraction from images'''

    def __init__(self, img=None, path=None, block_size=5):
        if path and not img:
            img = cv2.imread(path)

        #img = Preprocess().blur_image(img)
        self.block_size = block_size
        self.img_rgb = img.copy()
        self.img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        #self.img_ycbcr = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        self.img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        self.height, self.width, _ = img.shape
        self.Hxx, self.Hxy, self.Hyy = hessian_matrix(self.img_gray)
        #vector, self.hog = hog(self.img_gray, orientations=8, pixels_per_cell=(3, 3),
        #            cells_per_block=(1, 1), visualise=True)

        neighours = disk(25)
        self.entropy = entropy(self.img_gray, neighours)

    def to_samples(self, n='all'):
        """
        Extract samples from image.
        :param n: number of samples per image
        :return: list of samples
        """
        samples = []
        add_sample = lambda fts: samples.append(fts) if fts else None

        if n == 'all':
            for i in xrange(self.height):
                for j in xrange(self.width):
                    add_sample(self._get_features(i, j))
        else:
            cnt = 0
            while len(samples) < n:
                i, j = randint(0, self.height-1), randint(0, self.width-1)
                add_sample(self._get_features(i, j))

        return samples

    def _get_features(self, i, j):
        """
        Calculates vector of features for pixel given with i,j coordinate
        :param i:
        :param j:
        :return:
        """
        features = []
        margin = self.block_size/2

        for ii in xrange(i-margin, i+margin):
            for jj in xrange(j-margin, j+margin):
                if ii < 0 or jj < 0 or ii >= self.height or jj >= self.width:
                    return None
                else:
                    features.extend(self.img_rgb[ii, jj])
                    features.extend(self.img_hsv[ii, jj])
                    #features.extend(self.img_ycbcr[ii, jj])

                    features.append(self.Hxx[ii, jj])
                    features.append(self.Hxy[ii, jj])
                    features.append(self.Hyy[ii, jj])
                    features.append(self.entropy[ii, jj])

                    #features.append(self.hog[ii, jj])

        return features

    # def find_vegetation(self, classifier):
    #     """
    #     Extract vegetation mask from image using given classifier
    #     :param classifier:
    #     :return:
    #     """
    #     result = np.zeros((self.height, self.width))
    #     samples = []
    #     ij = []
    #
    #     for i in xrange(self.height):
    #         for j in xrange(self.width):
    #             fts = self._get_features(i, j)
    #             if fts:
    #                 samples.append(fts)
    #                 ij.append((i, j))
    #
    #     classified = classifier.predict(samples)
    #
    #     for point, label in zip(ij, classified):
    #         result[point] = label
    #
    #     return result
    def find_vegetation(self, classifier):
        """
        Extract vegetation mask from image using given classifier
        :param classifier:
        :return:
        """
        result = np.zeros((self.height, self.width))
        samples = []
        ij = []
        limit = 20000

        for i in xrange(self.height):
            for j in xrange(self.width):
                fts = self._get_features(i, j)
                if fts:
                    samples.append(fts)
                    ij.append((i, j))

                if len(samples) > limit:
                    classified = classifier.predict(samples)
                    for point, label in zip(ij, classified):
                        result[point] = label

                    samples = []
                    ij = []

        classified = classifier.predict(samples)
        for point, label in zip(ij, classified):
            result[point] = label

        samples = []
        ij = []
        return result

    def find_vegetation_slow(self, classifier):
        """
        Extract vegetation mask from image using given classifier
        :param classifier:
        :return:
        """
        result = np.zeros((self.height, self.width))
        for i in xrange(self.height):
            for j in xrange(self.width):
                fts = self._get_features(i, j)
                if fts:
                    result[i, j] = classifier.predict(fts)

        return result
