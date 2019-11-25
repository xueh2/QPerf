# coding=utf-8
import time
import imp
import os
import sys
import math
import copy
import random
import shutil

import numpy as np

import scipy as sp
from scipy.spatial import ConvexHull
from scipy.ndimage.morphology import binary_fill_holes
import scipy.misc

from skimage import io, transform
from skimage import measure

"""
cmr_ml_utils_geometry.py

This file contains utility functions for manipulation masks and contours
Main use case is the object detection and segmentation

"""

# ------------------------------------------------------------------------------
def smooth_contours (contour_x, contour_y, n_components=24, circularise=False, n_pts=2000):
    """ takes contour_x,contour_y the cartesian coordinates of a contour, 
        then procdues a smoothed more circular contour smoothed_contour_x,smoothed_contour_y"""

    try:

        if n_components is None:
            n_components=12 # slightly arbitary number,  but seems to work well
    
        npts=n_pts+1
        contour_pts = np.transpose(np.stack([contour_x,contour_y]))
    
        if circularise:
            # get the contour points that form a convex hull
            hull = sp.spatial.ConvexHull(contour_pts)
            to_sample = hull.vertices
        else:
            to_sample = range(0,len(contour_x))
           
        #wrap around cirlce
        to_sample = np.hstack([to_sample,to_sample[0]])
        sample_pts = contour_pts[to_sample,:]
                           
        # sample each curve at uniform distances according to arc length parameterisation
        dist_between_pts  = np.diff(sample_pts,axis=0)
        cumulative_distance = np.sqrt(dist_between_pts[:,0]**2 + dist_between_pts[:,1]**2)
        cumulative_distance = np.insert(cumulative_distance,0,0,axis=0)
        cumulative_distance = np.cumsum(cumulative_distance)
        cumulative_distance = cumulative_distance/cumulative_distance[-1]
        contour_x=np.interp(np.linspace(0,1,npts),cumulative_distance,sample_pts[:,0],period=360)
        contour_y=np.interp(np.linspace(0,1,npts),cumulative_distance,sample_pts[:,1],period=360)
        contour_x = contour_x[:-1]
        contour_y = contour_y[:-1]
        
        # smooth out contour by keeping the lowest nkeep Fourier components
        n = len (contour_x)
        nfilt=n-n_components-1
        f = np.fft.fft(contour_x)
        f[int(n/2+1-nfilt/2):int(n/2+nfilt/2)] = 0.0;
        smoothed_contour_x = np.abs(np.fft.ifft(f))
        f = np.fft.fft(contour_y)
        f[int(n/2+1-nfilt/2):int(n/2+nfilt/2)] = 0.0;
        smoothed_contour_y = np.abs(np.fft.ifft(f))

    except Exception as e:
        print("Error happened in smooth_contours ...", file=sys.stderr)
        print(e)
        sys.stderr.flush()

    return smoothed_contour_x, smoothed_contour_y

# ------------------------------------------------------------------------------

def extract_contours(preds, thres=0.75, smoothing=True, num_components_smoothing=24, circular=False, n_pts=2000):
    """Extract contours from segmentation mask or probability map

    Inputs:

        preds : [RO E1], input mask or probablity map
        thres : threshold to extract contours, a 2D marching cube extration is performed
        smoothing : True or False, if true, contours are smoothed
        num_components_smoothing : number of fft components kept after smoothing
        circular : True or False, if true, contours are kept to approx. circle

    Outputs:

        contours : a list of contours, every contour is a nx2 numpy array
    """

    try:
        contours = measure.find_contours(preds, thres)

        C_len = list()
    
        for n, contour in enumerate(contours):
            C_len.append(contours[n].shape[0])
    
        if smoothing:
            s_c = copy.deepcopy(contours)
            for n, contour in enumerate(contours):
                sc_x, sc_y = smooth_contours (contour[:, 0], contour[:, 1], n_components=num_components_smoothing, circularise=circular, n_pts=n_pts)

                s_c[n] = np.zeros((sc_x.shape[0], 2))
                s_c[n][:,0] = sc_x
                s_c[n][:,1] = sc_y

            contours = copy.deepcopy(s_c)

    except Exception as e:
        print("Error happened in extract_contours ...", file=sys.stderr)
        print(e)
        sys.stderr.flush()

    return contours, C_len

# ------------------------------------------------------------------------------
def extract_endo_epi_contours(preds, thres=0.75, smoothing=True, num_components_smoothing=24, circular=False, n_pts=2000):
    """Extract myocardium endo and epi contours from segmentation mask or probability map

    Inputs:

        preds : [RO E1], input mask or probablity map
        thres : threshold to extract contours, a 2D marching cube extration is performed
        smoothing : True or False, if true, contours are smoothed
        num_components_smoothing : number of fft components kept after smoothing
        circular : True or False, if true, contours are kept to approx. circle

    Outputs:

        endo : a nx2 numpy array for endo contour
        epi : a nx2 numpy array for epi contour
    """

    try:
        contours, C_len = extract_contours(preds, thres, smoothing, num_components_smoothing, circular, n_pts)

        num_c = len(contours)

        endo = None
        epi = None

        if num_c == 0:
            return endo, epi

        if num_c == 1:
            epi = contours[0]
            return endo, epi
    
        if num_c > 1:
            # find the longest contours as epi and the second longest as endo
            c_len = np.zeros([num_c])

            for n, contour in enumerate(contours):
                c_len[n] = C_len[n]
        
            c_ind = np.argsort(c_len)

            epi = contours[c_ind[-1]]
            endo = contours[c_ind[-2]]

    except Exception as e:
        print("Error happened in extract_endo_epi_contours ...", file=sys.stderr)
        print(e)
        sys.stderr.flush()

    return endo, epi

# ------------------------------------------------------------------------------

def extract_epi_contours(preds, thres=0.75, smoothing=True, num_components_smoothing=24, circular=False, n_pts=2000):
    """Extract myocardium epi contours from segmentation mask or probability map

    Inputs:

        preds : [RO E1], input mask or probablity map
        thres : threshold to extract contours, a 2D marching cube extration is performed
        smoothing : True or False, if true, contours are smoothed
        num_components_smoothing : number of fft components kept after smoothing
        circular : True or False, if true, contours are kept to approx. circle

    Outputs:

        epi : a nx2 numpy array for epi contour
    """

    try:

        contours, C_len = extract_contours(preds, thres, smoothing, num_components_smoothing, circular, n_pts)

        num_c = len(contours)

        epi = None

        if num_c == 0:
            return epi

        if num_c == 1:
            epi = contours[0]
            return epi

        if num_c > 1:
            # find the longest contours as epi
            c_len = np.zeros([num_c])

            for n, contour in enumerate(contours):
                c_len[n] = C_len[n]

            c_ind = np.argsort(c_len)

            print("Pick %d with len %d" % (c_ind[-1], c_len[c_ind[-1]]))

            epi = contours[c_ind[-1]]

    except Exception as e:
        print("Error happened in extract_epi_contours ...", file=sys.stderr)
        print(e)
        sys.stderr.flush()

    return epi

# ------------------------------------------------------------------------------

def extract_sector_contours(sectors, thres=0.95, smoothing=True, num_components_smoothing=36, circular=False, n_pts=2000):
    """Extract contours for every sector

    Inputs:

        sectors : [RO E1 3], mask for basal/medial/apex
        background is 0 
        basal/medial/apex: sector 1 - value 1, sector 2 - value 2 etc.

        thres : threshold to extract contours, a 2D marching cube extration is performed
        smoothing : True or False, if true, contours are smoothed
        num_components_smoothing : number of fft components kept after smoothing
        circular : True or False, if true, contours are kept to approx. circle

    Outputs:

        sector_C : a nx2x16 numpy array for sector contours
    """

    try:
        RO, E1, SLC = sectors.shape

        max_pts = 0

        basal = list()
        num_sectors_basal = int(np.max(sectors[:,:,0]))
        for i in np.arange(num_sectors_basal):
            mask = np.zeros((RO, E1))
            pts = np.where(np.squeeze(sectors[:,:,0])==i+1)
            mask[pts] = 1
            C = extract_epi_contours(mask, thres=thres, smoothing=True, num_components_smoothing=num_components_smoothing, circular=False, n_pts=n_pts)
            basal.append(C)
            if(C is not None):
                if(C.shape[0]>max_pts):
                    max_pts = C.shape[0]

        medial = list()
        num_sectors_medial = 0
        if(SLC>1):
            num_sectors_medial = int(np.max(sectors[:,:,1]))
            for i in np.arange(num_sectors_medial):
                mask = np.zeros((RO, E1))
                pts = np.where(np.squeeze(sectors[:,:,1])==i+1)
                mask[pts] = 1
                C = extract_epi_contours(mask, thres=thres, smoothing=True, num_components_smoothing=num_components_smoothing, circular=False, n_pts=n_pts)
                medial.append(C)
                if(C is not None):
                    if(C.shape[0]>max_pts):
                        max_pts = C.shape[0]

        apex = list()
        num_sectors_apex = 0
        if(SLC>2):
            num_sectors_apex = int(np.max(sectors[:,:,2]))
            for i in np.arange(num_sectors_apex):
                mask = np.zeros((RO, E1))
                pts = np.where(np.squeeze(sectors[:,:,2])==i+1)
                mask[pts] = 1
                C = extract_epi_contours(mask, thres=thres, smoothing=True, num_components_smoothing=num_components_smoothing, circular=False, n_pts=n_pts)
                apex.append(C)
                if(C is not None):
                    if(C.shape[0]>max_pts):
                        max_pts = C.shape[0]

        sector_C = np.zeros((max_pts, 2, int(num_sectors_basal+num_sectors_medial+num_sectors_apex)))-1

        for i in np.arange(num_sectors_basal):
            C = basal[i]
            if(C is not None):
                num = C.shape[0]
                sector_C[0:num, :, i] = C

        if(SLC>1):
            for i in np.arange(num_sectors_medial):
                C = medial[i]
                if(C is not None):
                    num = C.shape[0]
                    sector_C[0:num, :, i+num_sectors_basal] = C

        if(SLC>2):
            for i in np.arange(num_sectors_apex):
                C = apex[i]
                if(C is not None):
                    num = C.shape[0]
                    sector_C[0:num, :, i+num_sectors_medial+num_sectors_basal] = C

    except Exception as e:
        print("Error happened in extract_sector_contours ...", file=sys.stderr)
        print(e)
        sys.stderr.flush()

    return sector_C, basal, medial, apex

def extract_sector_contours_array(sectors, thres=0.95, smoothing=True, num_components_smoothing=36, circular=False, n_pts=2000):
    sectors_C, basal, medial, apex = extract_sector_contours(sectors, thres, smoothing, num_components_smoothing, circular, n_pts)
    return sectors_C
