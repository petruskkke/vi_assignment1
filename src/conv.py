from __future__ import division

import cv2
import logging


def conv_cv2(imgs, kernels, out_channels=-1):
    features = []
    for idx, img in enumerate(imgs):
        imfeatures = []
        for kernel in kernels:
            imfeatures.append(cv2.filter2D(img, out_channels, kernel))
        features.append(imfeatures)
        logging.debug('handled image: {0}'.format(idx))
    return features