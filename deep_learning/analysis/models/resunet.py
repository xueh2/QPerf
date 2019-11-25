import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .model_utils import *

"""
This NN is modified from original Unet paper, and combined good points from Vnet paper and ResUnet paper

Olaf Ronneberger, Philipp Fischer, Thomas Brox. U-Net: Convolutional Networks for Biomedical Image Segmentation. 
Medical Image Computing and Computer-Assisted Intervention (MICCAI), Springer, LNCS, Vol.9351: 234--241, 2015

Fausto Milletari, Nassir Navab, Seyed-Ahmad Ahmadi. V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation. MICCAI, 2016.

Zhengxin Zhang, Qingjie Liu, Yunhong Wang. Road Extraction by Deep Residual U-Net. IEEE GEOSCIENCE AND REMOTE SENSING LETTERS, 2017.

"""

class GadgetronResUnetInputBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, use_dropout=False, p=0.5, H=256, W=256, verbose=True):
        super(GadgetronResUnetInputBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)

        self.use_dropout = use_dropout

        H_conv1, W_conv1 = compute_output_size_2d(self.conv1, H, W)
        H_conv2, W_conv2 = compute_output_size_2d(self.conv2, H_conv1, W_conv1)

        self.H_out = H_conv2
        self.W_out = W_conv2

        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')

        self.num_conv2d = 2

        if verbose:
            print("    GadgetronResUnetInputBlock : input size (%d, %d, %d), output size (%d, %d, %d) --> (%d, %d, %d)" % (inplanes, H, W, planes, H_conv1, W_conv1, planes, H_conv2, W_conv2))

    def forward(self, x):
        out = self.conv2(self.relu(self.bn1(self.conv1(x))))

        if self.use_dropout:
            out = self.dp(out)

        return out

class GadgetronResUnetBasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, use_dropout=False, p=0.5, H=256, W=256, verbose=True):
        super(GadgetronResUnetBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)

        self.dp = nn.Dropout2d(p=p)

        self.use_dropout = use_dropout
        self.dropout_p = p
        self.stride = stride

        H_conv1, W_conv1 = compute_output_size_2d(self.conv1, H, W)
        H_conv2, W_conv2 = compute_output_size_2d(self.conv2, H_conv1, W_conv1)

        self.H_out = H_conv2
        self.W_out = W_conv2

        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')

        self.num_conv2d = 2

        if verbose:
            print("        GadgetronResUnetBasicBlock : input size (%d, %d, %d), output size (%d, %d, %d) --> (%d, %d, %d)" % (inplanes, H, W, planes, H_conv1, W_conv1, planes, H_conv2, W_conv2))

    def forward(self, x):       
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        if(x.shape[1] == out.shape[1]):
            out += x
            
        if self.use_dropout:
            out = self.dp(out)

        return out

class GadgetronResUnet_UpSample(nn.Module):
    def __init__(self, block, layers, in_ch, out_ch, bilinear=True, stride=1, use_dropout=False, p=0.5, H=256, W=256, verbose=True):
        super(GadgetronResUnet_UpSample, self).__init__()

        self.bilinear = bilinear
        if self.bilinear:
            self.up = []
        else:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)

        H_layer_in = 2*H
        W_layer_in = 2*W

        if verbose:
            print("        GadgetronResUnet_UpSample : input size (%d, %d, %d), upsampled size (%d, %d, %d)" % (in_ch, H, W, in_ch, H_layer_in, W_layer_in))

        self.layers = layers
        self.blocks = nn.Sequential()
        self.num_conv2d = 0
        for i in range(self.layers):
            module_name = "upsample %d" % i
            if i == 0:
                self.blocks.add_module(module_name, block(in_ch, out_ch, stride=stride, use_dropout=use_dropout, p=p, H=H_layer_in, W=W_layer_in, verbose=verbose))
            else:
                self.blocks.add_module(module_name, block(out_ch, out_ch, stride=stride, use_dropout=use_dropout, p=p, H=H_layer_in, W=W_layer_in, verbose=verbose))

            H_layer_in = self.blocks._modules[module_name].H_out
            W_layer_in = self.blocks._modules[module_name].W_out
            self.num_conv2d += self.blocks._modules[module_name].num_conv2d

        self.H_out = H_layer_in
        self.W_out = W_layer_in

        self.input = None

    def forward(self, x1):
        r"""
        x1: current input
        x2: from down sample layers
        """
        x2 = self.input

        if self.bilinear:
            x1 = nn.functional.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
        else:
            x1 = self.up(x1)

        if x1.shape[2] < x2.shape[2] or x1.shape[3] < x2.shape[3]:
            diffX = x2.size()[2] - x1.size()[2]
            diffY = x2.size()[3] - x1.size()[3]
            x1 = F.pad(x1, (diffX // 2, int(diffX / 2),
                            diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)

        x = self.blocks(x)

        return x

class GadgetronResUnet(nn.Module):

    def __init__(self, block, F0, inplanes, layers, layers_planes, use_dropout=False, p=0.5, bilinear=True, H=224, W=224, num_classes=10, verbose=True):
        r"""
        Implement the res-unet

        - Input:
        block: class name for basic building block, e.g. GadgetronResNetBasicBlock
        F0: number of channels in data
        inplanes: number of planes in input layer
        layers: e.g. [2 2 2], number of layers and number of basic blocks in each layer
        layers_planes: number of feature maps in each layer, [128 256 512]
        """

        super(GadgetronResUnet, self).__init__()

        self.use_dropout = use_dropout
        self.dropout_p = p
        self.bilinear = bilinear
        self.layers = layers
        self.layer_planes = layers_planes
        self.num_classes = num_classes
        
        if verbose:
            print("GadgetronResUnet : F0=%d, inplanes=%d" % (F0, inplanes))
            print("--" * 30)

        # --------------------------------------------------------------------
        # start up layer
        # --------------------------------------------------------------------
        self.input_layer = GadgetronResUnetInputBlock(F0, inplanes, stride=1, use_dropout=use_dropout, p=p, H=H, W=W, verbose=verbose)

        input_planes = inplanes
        self.num_conv2d = self.input_layer.num_conv2d

        if verbose:
            print("--" * 30)

        # --------------------------------------------------------------------
        # down sample layer
        # --------------------------------------------------------------------
        H_layer = self.input_layer.H_out
        W_layer = self.input_layer.W_out
        self.down_layers = nn.Sequential()
        for l in range(len(layers)):
            block_name = "Down layer %d" % l

            if verbose:
                print("    GadgetronResUnet, down layer %d:" % l)

            layer, H_layer_out, W_layer_out, num_layer_conv2d = self._make_down_layer(block, layers[l], inplanes, layers_planes[l], stride = 1, H_layer=H_layer, W_layer=W_layer, verbose=verbose)

            inplanes = layers_planes[l]
            H_layer, W_layer = H_layer_out, W_layer_out
            self.down_layers.add_module(block_name, layer)
            self.num_conv2d += num_layer_conv2d

        if verbose:
            print("--" * 30)

        # --------------------------------------------------------------------
        # bridge layer, still downsample along H and W, but do not increase planes
        # --------------------------------------------------------------------
        if verbose:
            print("    GadgetronResUnet, bridge layer (%d, %d, %d) --> (%d, %d, %d)" % (self.layer_planes[l], H_layer, W_layer, 
            self.layer_planes[l], H_layer_out, W_layer_out))
        self.bridge_layer, H_layer_out, W_layer_out, num_layer_conv2d = self._make_down_layer(block, layers[l], self.layer_planes[l], self.layer_planes[l], stride = 1, H_layer=H_layer, W_layer=W_layer, verbose=verbose)

        H_layer, W_layer = H_layer_out, W_layer_out
        self.num_conv2d += num_layer_conv2d

        if verbose:
            print("--" * 30)

        # --------------------------------------------------------------------
        # up sample layer
        # --------------------------------------------------------------------
        self.up_layers = nn.Sequential()
        for l in range(len(layers)):
            block_name = "Up layer %d" % l
            bl = len(layers) - l - 1

            if verbose:
                print("    GadgetronResUnet, up layer %d:" % l)

            if bl > 0:
                output_planes = layers_planes[bl-1]
            else:
                output_planes = input_planes

            layer, H_layer_out, W_layer_out, num_layer_conv2d = self._make_up_layer(block, layers[bl], 2*layers_planes[bl], output_planes, stride = 1, H_layer=H_layer, W_layer=W_layer, verbose=verbose)

            H_layer, W_layer = H_layer_out, W_layer_out
            self.up_layers.add_module(block_name, layer)
            self.num_conv2d += num_layer_conv2d

        if verbose:
                print("    GadgetronResUnet, up layer %d:" % (l+1))
        layer, H_layer_out, W_layer_out, num_layer_conv2d = self._make_up_layer(block, layers[0], 2*input_planes, input_planes, stride = 1, H_layer=H_layer, W_layer=W_layer, verbose=verbose)
        block_name = "Up layer %d" % (l+1)
        self.up_layers.add_module(block_name, layer)
        H_layer, W_layer = H_layer_out, W_layer_out
        self.num_conv2d += num_layer_conv2d

        if verbose:
            print("--" * 30)

        # --------------------------------------------------------------------
        # output layer, 1x1 conv
        # --------------------------------------------------------------------
        self.output_conv = nn.Conv2d(input_planes, num_classes, 1)
        if verbose:
            print("Output layer (%d, %d, %d) --> (%d, %d, %d)" % (input_planes, H_layer, W_layer, num_classes, H_layer, W_layer))
        if verbose:
            print("--" * 30)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_down_layer(self, block, layers, inplanes, planes, stride=1, H_layer=64, W_layer=64, verbose=True):
        layer = nn.Sequential()
        layer.add_module("downsample", nn.MaxPool2d(2))

        H_layer_in = H_layer/2
        W_layer_in = W_layer/2

        if verbose:
            print("        GadgetronResUnet, down layer (%d, %d) -> (%d, %d)" % (H_layer, W_layer, H_layer_in, W_layer_in))

        num_layer_conv2d = 0
        for i in range(layers):
            block_name = "ResBlock" + str(i)
            if i > 0:
                inplanes = planes
            layer.add_module(block_name, block(inplanes, planes, stride=stride, use_dropout=self.use_dropout, p=self.dropout_p, H=H_layer_in, W=W_layer_in, verbose=verbose))
            num_layer_conv2d += layer._modules[block_name].num_conv2d

        H_layer_out = layer._modules[block_name].H_out
        W_layer_out = layer._modules[block_name].W_out

        return layer, H_layer_out, W_layer_out, num_layer_conv2d

    def _make_up_layer(self, block, layers, inplanes, planes, stride=1, H_layer=64, W_layer=64, verbose=True):
        layer = GadgetronResUnet_UpSample(block, layers, inplanes, planes, self.bilinear, stride=stride, use_dropout=self.use_dropout, p=self.dropout_p, H=H_layer, W=W_layer, verbose=verbose)

        H_layer_out = layer.H_out
        W_layer_out = layer.W_out
        num_layer_conv2d = layer.num_conv2d

        return layer, H_layer_out, W_layer_out, num_layer_conv2d

    def forward(self, x):

        x_input = self.input_layer(x)
        num_layers = len(self.layers)

        # since the results from downsample layers are needed
        x_from_down_layers = []
        for l in range(len(self.layers)):
            if l==0:
                x = self.down_layers[l](x_input)
            else:
                x = self.down_layers[l](x)

            x_from_down_layers.append(x)

        x = self.bridge_layer(x)

        for l in range(num_layers+1):
            if l==num_layers:
                self.up_layers[l].input = x_input
            else:
                self.up_layers[l].input = x_from_down_layers[num_layers-l-1]

        x = self.up_layers(x)
        x = self.output_conv(x)

        return x

def GadgetronResUnet18(F0=3, inplanes=64, layers=[1,1,1], layers_planes=[128,256,512], use_dropout=False, p=0.5, H=224, W=224, C=2, verbose=True):
    """Constructs a GadgetronResUnet model.
    """
    model = GadgetronResUnet(GadgetronResUnetBasicBlock, F0, inplanes, layers, layers_planes, use_dropout=use_dropout, p=p, H=H, W=W, num_classes=C, verbose=verbose)
    return model
