from __future__ import division

import os
import cv2
import pickle
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from src.metrics import *

bins = 16

def color_histogram_vecs(fpath, vpath, rewrite=False):
    if os.path.exists(vpath) and os.path.isfile(vpath) and not rewrite:
        with open(vpath, 'rb') as f:
            vecs = pickle.load(f)

    else:
        imgs = []
        fnames = os.listdir(fpath)
        for fname in fnames:
            img = np.asarray(Image.open(os.path.join(fpath, fname)))
            imgs.append(img)
        
        vecs = []
        for idx, img in enumerate(imgs):
            channels = img.shape[2]
            w, h = img.shape[0], img.shape[1]
            imvec = []
            for c in range(0, channels):
                histogram = cv2.calcHist([img],[c],None,[bins],[0,256])
                histogram /= (w * h)
                imvec += list(histogram.reshape(bins))
            imvec = np.asarray(imvec)
            imbuffer = [fnames[idx], imvec]
            vecs.append(imbuffer)

        if os.path.exists(vpath) and os.path.isfile(vpath):
            os.remove(vpath)   

        with open(vpath, 'wb') as f:
            pickle.dump(vecs, f)

    return vecs


def query(qpath, dataset, topn=5):
    imgs = []
    fnames = os.listdir(qpath)
    for fname in fnames:
        img = np.asarray(Image.open(os.path.join(qpath, fname)))
        imgs.append(img)

    edtopnf, cstopnf = [], []
    for idx, img in enumerate(imgs):
        channels = img.shape[2]
        w, h = img.shape[0], img.shape[1]
        imvec = []
        for c in range(0, channels):
            histogram = cv2.calcHist([img],[c],None,[bins],[0,256])
            histogram /= (w * h)
            imvec += list(histogram.reshape(bins))
        imvec = np.asarray(imvec)

        edvecs, csvecs = [], []
        for data in dataset:
            edvecs.append(euclidean_distance(imvec, data[1]))
            csvecs.append(cosine_similarity(imvec, data[1]))
        
        edvecs, csvecs = np.array(edvecs), np.array(csvecs)
        edmaxv, csmaxv = np.max(edvecs), np.max(csvecs)

        edtopni, cstopni = [], []
        for tdx in range(0, topn):
            edmini, csmini = np.argmin(edvecs), np.argmin(csvecs)

            edtopni.append(edmini)
            cstopni.append(csmini)
            edvecs[edmini] = edmaxv
            csvecs[csmini] = csmaxv

        edtopnf.append([fnames[idx], [dataset[i][0] for i in edtopni]])
        cstopnf.append([fnames[idx], [dataset[i][0] for i in cstopni]])

    return edtopnf, cstopnf 


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logging.info('extract features by color histogram')

    topn = 5
    fpath = 'canvas/Assignment1_data/data'
    qpath = 'canvas/Assignment1_data/query'
    vpath = 'results/p4_imvec.pkl'

    data_vecs = color_histogram_vecs(fpath, vpath, rewrite=False)
    logging.info('feature vectors save in: {0}'.format(vpath))

    values = []
    for value in data_vecs:
        values.append(list(value[1]))
    values = np.asarray(values)

    for ndx in range(len(data_vecs)):
        data_vecs[ndx] = [data_vecs[ndx][0], values[ndx]]

    edtopnf, cstopnf = query(qpath, data_vecs, topn=topn)
    
    rows, cols = 5, topn + 1
    for _, topnf in enumerate([edtopnf]):
        idx = 0
        fig, ax = plt.subplots(rows, cols)

        for imtopnf in topnf:
            idx = int(imtopnf[0].split('.')[0])
            jdx = 1
            ax[idx, 0].axis('off')
            ax[idx, 0].imshow(np.asarray(Image.open(os.path.join(qpath, imtopnf[0]))))
            for topf in imtopnf[1]:
                ax[idx, jdx].axis('off')
                ax[idx, jdx].imshow( np.asarray(Image.open(os.path.join(fpath, topf))))
                jdx += 1
        plt.savefig('results/p4a1.jpg', bbox_inches='tight')
    logging.info('results save in: results/p4a1.jpg')
