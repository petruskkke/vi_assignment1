from __future__ import division

import os
import pickle
import logging
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from src.conv import conv_cv2
from src.metrics import *
from scripts.p2a import generate_kernels
from scripts.p2c import create_imvec


def fliter_bank_vecs(fpath, vpath, rewrite=False):
    if os.path.exists(vpath) and os.path.isfile(vpath) and not rewrite:
        with open(vpath, 'rb') as f:
            vecs = pickle.load(f)

    else:
        imgs = []
        fnames = os.listdir(fpath)
        for fname in fnames:
            img = np.asarray(Image.open(os.path.join(fpath, fname)).convert('L'))
            imgs.append(img)
        
        vecs = []
        for idx, img in enumerate(imgs):
            logging.info('handle image: {0}'.format(idx))
            imfeatures = conv_cv2(np.array([img]), kernels)
            imvec = create_imvec(imfeatures[0])
            imbuffer = [fnames[idx], imvec]
            vecs.append(imbuffer)
            
        if os.path.exists(vpath) and os.path.isfile(vpath):
            os.remove(vpath)   

        with open(vpath, 'wb') as f:
            pickle.dump(vecs, f)

    return vecs


def query(qpath, dataset, vmean, vvar, topn=5, hyper_params=[1,1,1,1]):
    imgs = []
    fnames = os.listdir(qpath)
    for fname in fnames:
        img = np.asarray(Image.open(os.path.join(qpath, fname)).convert('L'))
        imgs.append(img)

    edtopnf, cstopnf = [], []
    for idx, img in enumerate(imgs):
        imfeatures = conv_cv2(np.array([img]), kernels)
        imvec = create_imvec(imfeatures[0])

        imvec = (imvec - vmean) / vvar

        edvecs, csvecs = [], []
        for data in dataset:
            ed = hyper_params[0] * euclidean_distance(imvec[:12], data[1][:12]) \
                + hyper_params[1] * euclidean_distance(imvec[12:48], data[1][12:48]) \
                + hyper_params[2] * euclidean_distance(imvec[48:60], data[1][48:60]) \
                + hyper_params[3] * euclidean_distance(imvec[60:], data[1][60:])
            
            edvecs.append(ed)
            csvecs.append(cosine_similarity(imvec[:48], data[1][:48]))
        
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
    
    topn = 5
    kernels = generate_kernels(ksize=(32, 32))
    fpath = 'canvas/Assignment1_data/data'
    qpath = 'canvas/Assignment1_data/query'
    vpath = 'results/p3_imvec.pkl'

    data_vecs = fliter_bank_vecs(fpath, vpath, rewrite=False)
    logging.info('feature vectors save in: {0}'.format(vpath))

    values = []
    for value in data_vecs:
        values.append(list(value[1]))
    values = np.asarray(values)
    vmean = np.mean(values, axis=0)
    vvar = np.var(values, axis=0) + 1e-8
    values = (values - vmean) / vvar

    for ndx in range(len(data_vecs)):
        data_vecs[ndx] = [data_vecs[ndx][0], values[ndx]]

    hyper_params = [0.02, 0.1, 0.03, 2]
    edtopnf, cstopnf = query(qpath, data_vecs, vmean, vvar, topn=topn, hyper_params=hyper_params)

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
        plt.savefig('results/p3.jpg', bbox_inches='tight')
    logging.info('results save in: results/p3.jpg')

