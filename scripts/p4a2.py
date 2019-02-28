from __future__ import division

import cv2
import argparse
import logging
from src.nn import nnruner


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logging.info('extract features by deep learning')

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--train', action='store_true',
                        help='training an autoencoder')
    parser.add_argument('--pretrained', action='store_true',
                        help='use pretrained resnet152')

    args = parser.parse_args()

    nnruner(train=args.train, pretrained=args.pretrained)
    logging.info('deep learning result saves in \'results/p4a2_*.jpg\'')
