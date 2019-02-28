from __future__ import division

import os
import logging
import numpy as np
from PIL import Image

from src.conv import conv_cv2
from scripts.p2a import generate_kernels


def create_imvec(features):
    
    immean = []
    imvar = []
    for feature in features:
        immean.append(np.mean(feature))
        imvar.append(np.var(feature))
        imvec = np.asarray(immean + imvar)
    return imvec


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    kernels = generate_kernels(ksize=(32, 32))
    
    fpath = 'canvas/Assignment1_twoImg'
    fnames = os.listdir(fpath)

    imgs = []
    for fname in fnames:
        img = np.asarray(Image.open(os.path.join(fpath, fname)).convert('L'))
        imgs.append(img)

    features = conv_cv2(imgs, kernels)
    
    for ndx, imfeatures in enumerate(features):
        imvec = create_imvec(imfeatures)
        immean = imvec[:int(len(imvec) / 2)]
        imvar = imvec[int(len(imvec) / 2):]

        logging.info('image: {0}\n'
                     'max_mean: {1} with kernel: {2} \n' 
                     'max_var: {3} with kernel: {4}'.format(
                         fnames[ndx], np.max(immean), np.argmax(immean),
                         np.max(imvar), np.argmax(imvar)))
