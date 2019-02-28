from __future__ import division

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from src.conv import conv_cv2
from scripts.p2a import generate_kernels


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    rows, cols = 8, 6

    kernels = generate_kernels(ksize=(32, 32))

    fpath = 'canvas/Assignment1_twoImg'
    fnames = os.listdir(fpath)

    imgs = []
    for fname in fnames:
        img = np.asarray(Image.open(os.path.join(fpath, fname)))
        imgs.append(img)

    features = conv_cv2(imgs, kernels)
    
    for ndx, imfeatures in enumerate(features):
        idx, jdx = 0, 0
        fig, ax = plt.subplots(rows, cols)
        plt.subplots_adjust(wspace=0.2, hspace=0.2)
        
        for feature in imfeatures:
            if feature.max() - feature.min() != 0:
                feature = (feature - feature.min()) / (feature.max() - feature.min())

            ax[idx, jdx].axis('off')
            ax[idx, jdx].imshow(feature)

            jdx = (jdx + 1) % cols
            idx = (idx + 1) if jdx == 0  else idx 

        plt.savefig('results/p2b_{0}.jpg'.format(fnames[ndx].split('.')[0]), dpi=300, bbox_inches='tight')
    logging.info('result save in: results/p2b_*.jpg')