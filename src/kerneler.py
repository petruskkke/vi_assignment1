from __future__ import division

import cv2
import math
import numpy as np
from PIL import Image


def derivation(kernel, axis):
    # 0: x dim    1: y dim
    if axis == 0: 
        kernel = np.c_[kernel[:, 1:], kernel[:, -1]] - np.c_[kernel[:, 0], kernel[:, :-1]]
        kernel[:, -1] = 2 * kernel[:, -2] - kernel[:, -3]
        kernel[:, 0] = 2 * kernel[:, 1] - kernel[:, 2]
    elif axis == 1:
        kernel = np.r_[kernel[1:], kernel[-1:]] - np.r_[kernel[:1], kernel[:-1]]
        kernel[-1] = 2 * kernel[-2] - kernel[-3]
        kernel[0] = 2 * kernel[1] - kernel[2]
    return kernel


def gaussian_kernel(ksize, sigma=(1, 1), center=None):
    w, h = ksize[0], ksize[1]
    if center is None:
        center = (w / 2, h / 2)

    xdx_map = np.zeros((h, w), dtype=np.float64)
    ydx_map = np.zeros((h, w), dtype=np.float64)
    for hdx in range(0, h):
        for wdx in range(0, w):
            xdx_map[hdx, wdx] = wdx - center[0]
            ydx_map[hdx, wdx] = hdx - center[1]
    
    x2, y2 = xdx_map ** 2, ydx_map ** 2
    kernel = np.exp( -( (x2 / (2 * sigma[0] ** 2)) + (y2 / (2 * sigma[1] ** 2)) ) )
    kernel = 1 / (2 * np.pi * sigma[0] * sigma[1]) * kernel
    return kernel

# def gaussian_kernel(ksize, sigma=(1, 1), center=None):
#     w, h = ksize[0], ksize[1]
#     cov = np.mat([[sigma[1] ** 2, 0], [0, sigma[0] ** 2]])
#     if center is None:
#         center = (w - 1 / 2, h - 1 / 2)
#         print(center)

#     idx_map = np.zeros((h, w, 2), dtype=np.float32)
#     for hdx in range(0, h):
#         for wdx in range(0, w):
#             idx_map[hdx, wdx] = [hdx - center[1], wdx - center[0]]
    
#     kernel = np.zeros((h, w), dtype=np.float32)
#     for hdx in range(0, h):
#         for wdx in range(0, w):
#             t = idx_map[hdx, wdx, :]
#             k = np.exp(-0.5 * np.dot(np.dot(t.T, cov.I), t)) 
#             print(k)
#             k *= 1 / (np.sqrt((2 * np.pi) ** 2 * np.linalg.det(cov)))
#             kernel[hdx, wdx] = k
#         exit()
#     return kernel


def d1gaussian_kernel(ksize, alpha, sigma=(1, 1), center=None):
    g_kernel = gaussian_kernel(ksize, sigma=sigma, center=center)

    d1x = derivation(g_kernel, axis=0)    
    
    alpha = math.degrees(alpha)
    rmap = cv2.getRotationMatrix2D((ksize[1] / 2, ksize[0] / 2), alpha, 1)
    kernel = cv2.warpAffine(d1x, rmap, (ksize[1], ksize[0]))
    return kernel


def d2gaussian_kernel(ksize, alpha, sigma=(1, 1), center=None):
    g_kernel = gaussian_kernel(ksize, sigma=sigma, center=center)
    
    d1x = derivation(g_kernel, axis=0)    
    d2x = derivation(d1x, axis=0)    

    alpha = math.degrees(alpha)
    rmap = cv2.getRotationMatrix2D((ksize[1] / 2, ksize[0] / 2), alpha, 1)
    kernel = cv2.warpAffine(d2x, rmap, (ksize[1], ksize[0]))
    return kernel


def laplacian_gaussian_kernel(ksize, sigma=(1, 1), center=None):
    g_kernel = gaussian_kernel(ksize, sigma=sigma, center=center)

    d1x = derivation(g_kernel, axis=0)    
    d2x = derivation(d1x, axis=0)    
    d1y = derivation(g_kernel, axis=1) 
    d2y = derivation(d1y, axis=1) 

    kernel = d2x + d2y
    return kernel
