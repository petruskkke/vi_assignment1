from __future__ import division

import math
import logging
import numpy as np
import matplotlib.pyplot as plt

from src import kerneler

def draw_kernel(ax, k, idx, jdx, cols):
    # k = (k - k.min())/ (k.max() - k.min())
    ax[idx, jdx].axis('off')
    ax[idx, jdx].imshow(k)
    jdx = (jdx + 1) % cols
    idx = (idx + 1) if jdx == 0  else idx 
    return idx, jdx


def generate_kernels(ksize):
    kernels = []

    # gaussian kernel
    sigmas = [
        (1, 1), (math.sqrt(2), math.sqrt(2)), 
        (2, 2), (2 * math.sqrt(2), 2 * math.sqrt(2))]

    for sigma in sigmas:
        k = kerneler.gaussian_kernel(ksize, sigma)
        kernels.append(k)
    
    # laplacian gaussian kernel
    sigmas = [
        (math.sqrt(2), math.sqrt(2)), (2, 2), 
        (2 * math.sqrt(2), 2 * math.sqrt(2)), (4, 4), 
        (3 * math.sqrt(2), 3 * math.sqrt(2)), (6, 6), 
        (6 * math.sqrt(2), 6 * math.sqrt(2)), (12, 12)]

    for sigma in sigmas:
        k = kerneler.laplacian_gaussian_kernel(ksize, sigma)
        kernels.append(k)

    # d1gaussian and d2gaussian kernel
    sigmas = [
        (1, 3), (math.sqrt(2), 3 * math.sqrt(2)), (2, 6)]
    alphas = [
        0, np.pi / 6, np.pi / 3, np.pi / 2, np.pi * 2 / 3, np.pi * 5 / 6]
    
    for sigma in sigmas:
        for alpha in alphas:
            k = kerneler.d1gaussian_kernel(ksize, alpha, sigma)
            kernels.append(k)
            
    for sigma in sigmas:
        for alpha in alphas:
            k = kerneler.d2gaussian_kernel(ksize, alpha, sigma)
            kernels.append(k)
            
    return kernels

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    rows, cols = 6, 8
    fig, ax = plt.subplots(rows, cols)

    idx, jdx = 0, 0 
    ksize = (32, 32)
    kernels = generate_kernels(ksize)

    for k in kernels:
        idx, jdx = draw_kernel(ax, k, idx, jdx, cols=cols)
    plt.savefig('results/p2a.jpg', bbox_inches='tight')
    logging.info('result save in: results/p2a.jpg')