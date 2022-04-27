import cv2
import os
import numpy as np
from tqdm import tqdm
from src.matrix_ops import ops

from functools import *

# Define your smoothing functions here
def mean_smoothing(batch, filter_size=(5,5)):
    # An example smoothing function, your function should look similar
    '''
        Refer to this link for more information on the smoothing function used (also contains all the other smoothing functions)
        https://opencv24-python-tutorials.readthedocs.io/en/stable/py_tutorials/py_imgproc/py_filtering/py_filtering.html
        @params <np.array> shape==(batch_size, 1, n, n) inpt, the input that needs to be smoothed
        @params <tuple> filter_size, a tuples with two entries that define the size of the filter
        @returns <np.array> shape==(batch_size, 1, n, n) a smoothed out input
    '''
    smoothed_batch = np.zeros(batch.shape)
    for idx, sample in enumerate(batch):
        blurred = cv2.blur(sample[0], filter_size)
        smoothed_batch[idx, 0, :, :] = blurred
    
    return smoothed_batch

def gaussian_smoothing(batch, filter_size=(17,17), sigma=7):
    '''
        A function that takes in a batch of images and smoothes them using a gaussian filter

        @params <np.array> shape==(batch_size, 1, n, n) inpt, the input that needs to be smoothed
        @params <tuple> filter_size, a tuples with two entries that define the size of the filter
        @returns <np.array> shape==(batch_size, 1, n, n) a smoothed out input
    '''
    # An example smoothing function, your function should look similar
    smoothed_batch = np.zeros(batch.shape)
    for idx, sample in enumerate(batch):
        blurred = cv2.GaussianBlur(sample[0], filter_size, sigma)
        smoothed_batch[idx, 0, :, :] = blurred
    
    return smoothed_batch




def upscale(conf):
    return partial(gaussian_smoothing, filter_size=conf['configurations']['kernel_size'],
                  sigma=conf['configurations']['sigma'])
