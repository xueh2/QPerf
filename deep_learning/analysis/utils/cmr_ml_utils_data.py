# coding=utf-8
import time
import imp
import os
import sys
import math
import copy
import random
import shutil
import random
import numpy as np
import scipy as sp
import scipy.misc
from skimage import io, transform
from skimage import measure
from PIL import Image
from glob import glob
import sklearn
import logging

"""
This file contains utility functions for manipulating data and controlling training
"""

# ------------------------------------------------------------------------------
def chunk(xs, n):    
    """Given a list of samples, split them to n chunks
    The main usage is for k-fold cross-validation

    Usage:
        chunks = chunk(range(len(perf_dataset)), n=4)
        listified_chunks = list(chunks)

        This will split all data samples to 4 chunks after random shuffling

        listified_chunks will have indexes for every chunk
    """
    ys = list(xs)
    random.shuffle(ys)
    size = len(ys) // n
    leftovers= ys[size*n:]
    for c in range(n):
        if leftovers:
           extra= [ leftovers.pop() ] 
        else:
           extra= []
        yield ys[c*size:(c+1)*size] + extra

# ------------------------------------------------------------------------------
def get_k_fold_training_validation(chunks, val_chunk=0):    
    """Given the chunks, return the idx for training and validation samples

    Inputs:
        val_chunk : chunk id of validation set; all other chunks will be used for training    

    Usage:
        chunks = chunk(range(len(perf_dataset)), n=4)
        training_idx, validation_idx = get_k_fold_training_validation(chunks, val_chunk)
    """
    listified_chunks = list(chunks)
    val_idxs = listified_chunks[val_chunk]

    train_idxs = list()
    if val_chunk>0:
        train_idxs.append(listified_chunks[0:val_chunk])

    train_idxs.append(listified_chunks[val_chunk+1:])
    train_idxs = [item for sublist in train_idxs for item in sublist ]
    train_idxs = [item for sublist in train_idxs for item in sublist]

    return train_idxs, val_idxs