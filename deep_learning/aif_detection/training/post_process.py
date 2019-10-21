import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import scipy
from scipy import ndimage

def dice_coeff(y_pred, y):
    a = y_pred.view(-1).float()
    b = y.view(-1).float()
    inter = torch.dot(a, b) + 0.0001
    union = torch.sum(a) + torch.sum(b) + 0.0001
    return 2*inter.float()/union.float()

def centroid_diff(pred, mask):
    y,x = np.nonzero(pred.cpu().numpy())
    pred_centroid = np.array([x.mean(), y.mean()])
    
    y,x = np.nonzero(mask.cpu().numpy())
    mask_centroid = np.array([x.mean(), y.mean()])
    return (pred_centroid, mask_centroid, np.sqrt(np.sum((pred_centroid - mask_centroid)**2)))

# probs - Output of neural network
# device - Device for variables to be run on.
# params - Extra paremeters if needed (e.g. aif_moco_echo1)
def adaptive_thresh(probs, device, p_thresh=0.5, params=None):
    # Try regular adaptive thresholding first
    p_thresh_max  = 0.96 # <-- Should not be too close to 1 to ensure while loop does not go over.
                         # Note that p_thresh_max = 0.99 was used for abstract evaluation.
    p_thresh_incr = 0.01

    cpu_device = torch.device('cpu')

    probs = probs.to(device=cpu_device)

    RO = probs.shape[0]
    E1 = probs.shape[1]

    number_of_blobs = float("inf")
    blobs = np.zeros((RO,E1))
    while number_of_blobs > 1 and p_thresh < p_thresh_max:
        mask = (probs > torch.max(probs) * p_thresh).float()
        blobs, number_of_blobs = ndimage.label(mask)
        p_thresh += p_thresh_incr # <-- Note this line can lead to float drift.

    if(number_of_blobs == 1):
        return mask

    ## If we are here then we cannot isolate a singular blob as the LV.
    ## Select the largest blob as the final mask.
    biggest_blob = (0, torch.zeros(RO,E1))

    for i in range(number_of_blobs):
        #one_blob = torch.tensor( (blobs == i+1).astype(np.float32) )
        one_blob = torch.tensor((blobs == i+1).astype(int), dtype=torch.uint8)
        if device is None:
            one_blob = one_blob.to(torch.uint8).cuda() 
        else:
            one_blob = one_blob.to(device=cpu_device, dtype=torch.uint8) 

        area = torch.sum(one_blob)
        if(area > biggest_blob[0]):
            biggest_blob = (area, one_blob)

    res = biggest_blob[1].to(device=device)

    return res
