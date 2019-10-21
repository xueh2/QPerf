# coding=utf-8
import time
import imp
import os
import sys
import math
import copy
import random
import shutil

import matplotlib.pyplot as plt
from matplotlib import animation, rc
animation.rcParams['animation.writer'] = 'ffmpeg'
plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

from IPython.display import display, clear_output, HTML, Image

import numpy as np

import scipy as sp
from scipy.spatial import ConvexHull
from scipy.ndimage.morphology import binary_fill_holes
import scipy.misc

from skimage import io, transform
from skimage import measure

from PIL import Image
from glob import glob
import sklearn
import logging

"""
This file contains utility functions for plotting/saving images and contours
"""

# ------------------------------------------------------------------------------
def save_as_image(a, img_name='test', img_dir='./DebugOutput'):
    """Save image as tif files

    Inputs:
        a : [RO E1 C N], C can be 1 for intensity image or 3 for turecolor image
        img_name : image name, appended with indexes
        img_dir : directory to save images
    """
    RO, E1, C, N = a.shape
      
    for n in range(N):
        filename = os.path.join(img_dir, img_name + str(n) + '.tif')
        if C==3:
            plt.imsave(filename, a[:,:,:,n])
            continue
        if C==1:
            plt.imsave(filename, a[:,:,0,n])
            continue
        pass

# ------------------------------------------------------------------------------

def plot_image_array(im, columns=4, figsize=[32, 32], cmap='gray'):
    """Plot image array as a panel of images
    """
    fig=plt.figure(figsize=figsize)
    plt.set_cmap(cmap)
    
    if(len(im.shape)==3):
        RO, E1, N = im.shape
    else:
        RO, E1 = im.shape
        N = 1

    rows = np.ceil(N/columns)
    for i in range(1, N+1):
        fig.add_subplot(rows, columns, i)
        if(len(im.shape)==3):
            plt.imshow(im[:,:,i-1])
        else:
            plt.imshow(im)
    plt.show()
    
    return fig

# ------------------------------------------------------------------------------

def play_image_array_movie(im, fig, ax, interval = 30, cmap='gray', interpolation='nearest'):
    """Plot image array as movie

    Inputs: 
        im : [RO, E1, N], image array
        fig, ax : figure and axis handle
        interval : in ms, plotting interval

    Usage:

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[8, 8])
        anim = play_image_array(Gd, fig, ax, interval=30)
        HTML(anim) 
    """
    
    ax.xlim = (0, im.shape[1])
    ax.ylim = (0, im.shape[2])
    ax.set_xticks([])
    ax.set_yticks([])
    img = ax.imshow(im[:, :, 0].T, cmap=cmap)
    img.set_interpolation(interpolation)
    
    def animate(i): 
        img.set_data(im[:, :, i].T)
        clear_output(wait=True)
        sys.stdout.flush()
        return (img,)

    # call the animator. blit=True means only re-draw the parts that have changed.
    # *interval* draws a new frame every *interval* milliseconds.
    anim = animation.FuncAnimation(fig, animate, frames=im.shape[-1], interval=interval, blit=False)
    return anim.to_html5_video()

# ------------------------------------------------------------------------------

def plot_contours_on_image(im, contours, ax, cmap='gray', interpolation='nearest', linewidth=2, linestyle='solid'):
    """Plot contours on image

    Inputs:
        ax : axis of the plotting
        contours : a list of contours, every contour is a Nx2 numpy array
    """

    ax.imshow(im, interpolation=interpolation, cmap=cmap)
    for n, contour in enumerate(contours):
        ax.plot(contour[:, 1], contour[:, 0], linewidth=linewidth, linestyle=linestyle)
        pts = np.mean(contour, axis=0)
        C_str = 'C %d' % (n+1)
        ax.text(pts[1], pts[0], C_str)
        
    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])