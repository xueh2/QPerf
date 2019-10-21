import torch.nn as nn
import math

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def compute_output_size_2d(m, H, W):
    """
    Compute output size for conv and pooling layer

    Input:
    - m: module, can be Conv2d or MaxPool2d or AvePool2d
    - H, W: inpute 2d image size [N C H W]

    Output:
    - H_out, W_out: output size
    """

    H_out = H
    W_out = W

    if isinstance(m, nn.Conv2d):
        kernel_size = m.kernel_size
        stride = m.stride
        padding = m.padding
        dilation = m.dilation

        H_out = int( (H + 2*padding[0] - dilation[0]*(kernel_size[0]-1) -1)/stride[0] ) + 1
        W_out = int( (W + 2*padding[1] - dilation[1]*(kernel_size[1]-1) -1)/stride[1] ) + 1
    elif isinstance(m, nn.modules.pooling.AvgPool2d):
        kernel_size = m.kernel_size
        stride = m.stride
        padding = m.padding

        H_out = int( (H + 2*padding - kernel_size)/stride ) + 1
        W_out = int( (W + 2*padding - kernel_size)/stride ) + 1
    else:
        raise Exception('Unsupport module type ... ')

    return H_out, W_out

def compute_output_size_maxpool2d(m, H, W):
    """
    Compute output size for max pooling layer

    Input:
    - m: module, can be Conv2d or MaxPool2d or AvePool2d
    - H, W: inpute 2d image size [N C H W]

    Output:
    - H_out, W_out: output size
    """

    H_out = H
    W_out = W

    kh = m.kernel_size[0]
    kw = m.kernel_size[1]
    ph = m.padding[0]
    pw = m.padding[1]

    if m.stride[0] == 2:
        H_out = int( (H + 2*ph - kh)/2 ) +1
        W_out = int( (W + 2*pw - kw)/2 ) +1
    else:
        H_out = H + 2*ph - kh + 1
        W_out = W + 2*pw - kw + 1

    return H_out, W_out

    