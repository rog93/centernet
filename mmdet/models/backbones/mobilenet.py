import math
import logging
import numpy as np
from os.path import join
import os

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from .dla import DLAMain,BasicBlock,DLAUp,IDAUp

# from .DCNv2.dcn_v2 import DCN
from ..registry import BACKBONES

logger = logging.getLogger(__name__)

def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]

def load_state_dict(checkpoint_path, use_ema=False):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict_key = 'state_dict'
        if state_dict_key and state_dict_key in checkpoint:
            if use_ema and 'state_dict_ema' in checkpoint:
                state_dict_key = 'state_dict_ema'
            state_dict = checkpoint[state_dict_key]
        else:
            state_dict = checkpoint
        #if state_dict_key and state_dict_key in checkpoint:
        if 1:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                # strip `module.` prefix
                name = k[7:] if k.startswith('module') else k
                if 'fc.weight' == name:
                    classifier_weight = v[:5]
                    new_state_dict[name] = classifier_weight[:]
               
                elif 'fc.bias' == name:
                    classifier_bias = v[:5]
                    new_state_dict[name] = classifier_bias[:]
                
                else:
                    new_state_dict[name] = v
                # del new_state_dict['fc.weight']
                # del new_state_dict['fc.bias']
            state_dict = new_state_dict
        else:
            state_dict = checkpoint
        logging.info("Loaded {} from checkpoint '{}'".format(state_dict_key, checkpoint_path))
        return state_dict
    else:
        logging.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


def load_checkpoint(model, checkpoint_path, use_ema=False):
    state_dict = load_state_dict(checkpoint_path, use_ema)
    model.load_state_dict(state_dict, strict=False)

class ConvBnRelu(nn.Module):
    def __init__(self, inplanes, first_planes, outplanes, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,):
        super(ConvBnRelu, self).__init__()

        self.conv = nn.Conv2d(inplanes, first_planes, kernel_size=3, padding=1, bias=False)
        self.bn = norm_layer(first_planes)

        self.conv_h = nn.Conv2d(inplanes, first_planes, kernel_size=(1,3), padding=(0,1), bias=False)
        self.bn_h = norm_layer(first_planes)

        self.conv_v = nn.Conv2d(inplanes, first_planes, kernel_size=(3,1), padding=(1,0), bias=False)
        self.bn_v = norm_layer(first_planes)

        self.act = act_layer(inplace=True)

    def forward(self, x):

        x1 = self.conv(x)
        x1 = self.bn(x1)
    
        x2 = self.conv_h(x)
        x2 = self.bn_h(x2)

        x3 = self.conv_v(x)
        x3 = self.bn_v(x3)

        x = x1+x2+x3
        x = self.act(x)
        return x


class ConvBnRelu1x1(nn.Module):
    def __init__(self, inplanes, first_planes, outplanes, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,):
        super(ConvBnRelu1x1, self).__init__()

        self.conv = nn.Conv2d(inplanes, first_planes, kernel_size=1, padding=0, bias=False)
        self.bn = norm_layer(first_planes)

        # self.conv_h = nn.Conv2d(inplanes, first_planes, kernel_size=(1,3), padding=(0,1), bias=False)
        # self.bn_h = norm_layer(first_planes)

        # self.conv_v = nn.Conv2d(inplanes, first_planes, kernel_size=(3,1), padding=(1,0), bias=False)
        # self.bn_v = norm_layer(first_planes)

        self.act = act_layer(inplace=True)

    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        # x2 = self.conv_dw_(x)
        # x2 = self.bn_dw_h(x2)

        # x3 = self.conv_dw_v(x)
        # x3 = self.bn_dw_v(x3)

        # x = x1+x2+x3

        return x


class DlaBlock1(nn.Module):
    def __init__(self, inplanes, first_planes, outplanes, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,):
        super(DlaBlock1, self).__init__()

        self._1_conv1 = nn.Conv2d(inplanes, first_planes, stride = 2, kernel_size=3, padding=1, bias=False)
        self._1_bn1 = norm_layer(first_planes)

        self._1_conv1_h = nn.Conv2d(inplanes, first_planes,stride = 2, kernel_size=(1,3), padding=(0,1), bias=False)
        self._1_bn1_h = norm_layer(first_planes)

        self._1_conv1_v = nn.Conv2d(inplanes, first_planes, stride = 2, kernel_size=(3,1), padding=(1,0), bias=False)
        self._1_bn1_v = norm_layer(first_planes)

        self._1_act1 = act_layer(inplace=True)


        self._1_conv2 = nn.Conv2d(first_planes, first_planes, stride = 1, kernel_size=3, padding=1, bias=False)
        self._1_bn2 = norm_layer(first_planes)

        self._1_conv2_h = nn.Conv2d(first_planes, first_planes,stride = 1, kernel_size=(1,3), padding=(0,1), bias=False)
        self._1_bn2_h = norm_layer(first_planes)

        self._1_conv2_v = nn.Conv2d(first_planes, first_planes, stride = 1, kernel_size=(3,1), padding=(1,0), bias=False)
        self._1_bn2_v = norm_layer(first_planes)

        self._1_act2 = act_layer(inplace=True)

        self._1_pool = nn.MaxPool2d(stride=2, kernel_size=2)
        self._1_pool_conv = nn.Conv2d(inplanes, first_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self._1_pool_bn = norm_layer(first_planes)

        self._2_conv1 = nn.Conv2d(first_planes, first_planes, stride = 1, kernel_size=3, padding=1, bias=False)
        self._2_bn1 = norm_layer(first_planes)

        self._2_conv1_h = nn.Conv2d(first_planes, first_planes, stride = 1, kernel_size=(1,3), padding=(0,1), bias=False)
        self._2_bn1_h = norm_layer(first_planes)

        self._2_conv1_v = nn.Conv2d(first_planes, first_planes, stride = 1, kernel_size=(3,1), padding=(1,0), bias=False)
        self._2_bn1_v = norm_layer(first_planes)
        self._2_act1 = act_layer(inplace=True)

        self._2_conv2 = nn.Conv2d(first_planes, first_planes, stride = 1, kernel_size=3, padding=1, bias=False)
        self._2_bn2 = norm_layer(first_planes)

        self._2_conv2_h = nn.Conv2d(first_planes, first_planes,stride = 1, kernel_size=(1,3), padding=(0,1), bias=False)
        self._2_bn2_h = norm_layer(first_planes)

        self._2_conv2_v = nn.Conv2d(first_planes, first_planes, stride = 1, kernel_size=(3,1), padding=(1,0), bias=False)
        self._2_bn2_v = norm_layer(first_planes)

        self._2_act2 = act_layer(inplace=True)

        #self._2_conv_proj = nn.Conv2d(inplanes + first_planes * 2, first_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self._2_conv_proj = nn.Conv2d(first_planes, first_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self._2_bn_proj = norm_layer(first_planes)
        self._2_act_proj = act_layer(inplace=True)

    def forward(self, x):
        r = x

        r = self._1_pool(r)
        #r1 = r
        r = self._1_pool_conv(r)
        r = self._1_pool_bn(r)

        x1 = self._1_conv1(x)
        x1 = self._1_bn1(x1)

        x2 = self._1_conv1_h(x)
        x2 = self._1_bn1_h(x2)

        x3 = self._1_conv1_v(x)
        x3 = self._1_bn1_v(x3)

        x = x1+x2+x3
        x = self._1_act1(x)

        x1 = self._1_conv2(x)
        x1 = self._1_bn2(x1)

        x2 = self._1_conv2_h(x)
        x2 = self._1_bn2_h(x2)

        x3 = self._1_conv2_v(x)
        x3 = self._1_bn2_v(x3)

        x = x1+x2+x3

        x = x + r
        x = self._1_act2(x)

        r2 = x

        x1 = self._2_conv1(x)
        x1 = self._2_bn1(x1)

        x2 = self._2_conv1_h(x)
        x2 = self._2_bn1_h(x2)

        x3 = self._2_conv1_v(x)
        x3 = self._2_bn1_v(x3)

        x = x1+x2+x3
        x = self._2_act1(x)

        x1 = self._2_conv2(x)
        x1 = self._2_bn2(x1)

        x2 = self._2_conv2_h(x)
        x2 = self._2_bn2_h(x2)

        x3 = self._2_conv2_v(x)
        x3 = self._2_bn2_v(x3)
        
        x = x1+x2+x3

        x = x + r2
        x = self._2_act2(x)

        #x = torch.cat([r1, r2, x], 1)
        #x = r2 + x
        x = self._2_conv_proj(x)
        x = self._2_bn_proj(x)
        x = self._2_act_proj(x)

        return x

class DlaBlock2(nn.Module):
    def __init__(self, inplanes, first_planes, outplanes, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,):
        super(DlaBlock2, self).__init__()

        self._1_conv1 = nn.Conv2d(inplanes, first_planes, stride = 2, kernel_size=3, padding=1, bias=False)
        self._1_bn1 = norm_layer(first_planes)

        self._1_conv1_h = nn.Conv2d(inplanes, first_planes,stride = 2, kernel_size=(1,3), padding=(0,1), bias=False)
        self._1_bn1_h = norm_layer(first_planes)

        self._1_conv1_v = nn.Conv2d(inplanes, first_planes, stride = 2, kernel_size=(3,1), padding=(1,0), bias=False)
        self._1_bn1_v = norm_layer(first_planes)

        self._1_act1 = act_layer(inplace=True)


        self._1_conv2 = nn.Conv2d(first_planes, first_planes, stride = 1, kernel_size=3, padding=1, bias=False)
        self._1_bn2 = norm_layer(first_planes)

        self._1_conv2_h = nn.Conv2d(first_planes, first_planes,stride = 1, kernel_size=(1,3), padding=(0,1), bias=False)
        self._1_bn2_h = norm_layer(first_planes)

        self._1_conv2_v = nn.Conv2d(first_planes, first_planes, stride = 1, kernel_size=(3,1), padding=(1,0), bias=False)
        self._1_bn2_v = norm_layer(first_planes)

        self._1_act2 = act_layer(inplace=True)

        self._1_pool = nn.MaxPool2d(stride=2, kernel_size=2)
        self._1_pool_conv = nn.Conv2d(inplanes, first_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self._1_pool_bn = norm_layer(first_planes)

        self._2_conv1 = nn.Conv2d(first_planes, first_planes, stride = 1, kernel_size=3, padding=1, bias=False)
        self._2_bn1 = norm_layer(first_planes)

        self._2_conv1_h = nn.Conv2d(first_planes, first_planes, stride = 1, kernel_size=(1,3), padding=(0,1), bias=False)
        self._2_bn1_h = norm_layer(first_planes)

        self._2_conv1_v = nn.Conv2d(first_planes, first_planes, stride = 1, kernel_size=(3,1), padding=(1,0), bias=False)
        self._2_bn1_v = norm_layer(first_planes)
        self._2_act1 = act_layer(inplace=True)

        self._2_conv2 = nn.Conv2d(first_planes, first_planes, stride = 1, kernel_size=3, padding=1, bias=False)
        self._2_bn2 = norm_layer(first_planes)

        self._2_conv2_h = nn.Conv2d(first_planes, first_planes,stride = 1, kernel_size=(1,3), padding=(0,1), bias=False)
        self._2_bn2_h = norm_layer(first_planes)

        self._2_conv2_v = nn.Conv2d(first_planes, first_planes, stride = 1, kernel_size=(3,1), padding=(1,0), bias=False)
        self._2_bn2_v = norm_layer(first_planes)

        self._2_act2 = act_layer(inplace=True)

        #self._2_conv_proj = nn.Conv2d(inplanes + first_planes * 2, first_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self._2_conv_proj = nn.Conv2d(first_planes, first_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self._2_bn_proj = norm_layer(first_planes)
        self._2_act_proj = act_layer(inplace=True)


        self._3_conv1 = nn.Conv2d(first_planes, first_planes, stride = 1, kernel_size=3, padding=1, bias=False)
        self._3_bn1 = norm_layer(first_planes)

        self._3_conv1_h = nn.Conv2d(first_planes, first_planes,stride = 1, kernel_size=(1,3), padding=(0,1), bias=False)
        self._3_bn1_h = norm_layer(first_planes)

        self._3_conv1_v = nn.Conv2d(first_planes, first_planes, stride = 1, kernel_size=(3,1), padding=(1,0), bias=False)
        self._3_bn1_v = norm_layer(first_planes)

        self._3_act1 = act_layer(inplace=True)


        self._3_conv2 = nn.Conv2d(first_planes, first_planes, stride = 1, kernel_size=3, padding=1, bias=False)
        self._3_bn2 = norm_layer(first_planes)

        self._3_conv2_h = nn.Conv2d(first_planes, first_planes,stride = 1, kernel_size=(1,3), padding=(0,1), bias=False)
        self._3_bn2_h = norm_layer(first_planes)

        self._3_conv2_v = nn.Conv2d(first_planes, first_planes, stride = 1, kernel_size=(3,1), padding=(1,0), bias=False)
        self._3_bn2_v = norm_layer(first_planes)

        self._3_act2 = act_layer(inplace=True)

        self._4_conv1 = nn.Conv2d(first_planes, first_planes, stride = 1, kernel_size=3, padding=1, bias=False)
        self._4_bn1 = norm_layer(first_planes)

        self._4_conv1_h = nn.Conv2d(first_planes, first_planes, stride = 1, kernel_size=(1,3), padding=(0,1), bias=False)
        self._4_bn1_h = norm_layer(first_planes)

        self._4_conv1_v = nn.Conv2d(first_planes, first_planes, stride = 1, kernel_size=(3,1), padding=(1,0), bias=False)
        self._4_bn1_v = norm_layer(first_planes)
        self._4_act1 = act_layer(inplace=True)

        self._4_conv2 = nn.Conv2d(first_planes, first_planes, stride = 1, kernel_size=3, padding=1, bias=False)
        self._4_bn2 = norm_layer(first_planes)

        self._4_conv2_h = nn.Conv2d(first_planes, first_planes,stride = 1, kernel_size=(1,3), padding=(0,1), bias=False)
        self._4_bn2_h = norm_layer(first_planes)

        self._4_conv2_v = nn.Conv2d(first_planes, first_planes, stride = 1, kernel_size=(3,1), padding=(1,0), bias=False)
        self._4_bn2_v = norm_layer(first_planes)

        self._4_act2 = act_layer(inplace=True)

        #self._4_conv_proj = nn.Conv2d(inplanes + first_planes * 3, first_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self._4_conv_proj = nn.Conv2d(first_planes, first_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self._4_bn_proj = norm_layer(first_planes)
        self._4_act_proj = act_layer(inplace=True)

    def forward(self, x):
        r = x
        r = self._1_pool(r)
        #r1 = r
        r = self._1_pool_conv(r)
        r = self._1_pool_bn(r)

        x1 = self._1_conv1(x)
        x1 = self._1_bn1(x1)

        x2 = self._1_conv1_h(x)
        x2 = self._1_bn1_h(x2)

        x3 = self._1_conv1_v(x)
        x3 = self._1_bn1_v(x3)

        x = x1+x2+x3
        x = self._1_act1(x)

        x1 = self._1_conv2(x)
        x1 = self._1_bn2(x1)

        x2 = self._1_conv2_h(x)
        x2 = self._1_bn2_h(x2)

        x3 = self._1_conv2_v(x)
        x3 = self._1_bn2_v(x3)

        x = x1+x2+x3

        x = x + r
        x = self._1_act2(x)

        r2 = x

        x1 = self._2_conv1(x)
        x1 = self._2_bn1(x1)

        x2 = self._2_conv1_h(x)
        x2 = self._2_bn1_h(x2)

        x3 = self._2_conv1_v(x)
        x3 = self._2_bn1_v(x3)

        x = x1+x2+x3
        x = self._2_act1(x)

        x1 = self._2_conv2(x)
        x1 = self._2_bn2(x1)

        x2 = self._2_conv2_h(x)
        x2 = self._2_bn2_h(x2)

        x3 = self._2_conv2_v(x)
        x3 = self._2_bn2_v(x3)
        
        x = x1+x2+x3

        x = x + r2
        x = self._2_act2(x)

        #x = torch.cat([r1, r2, x], 1)

        x = self._2_conv_proj(x)
        x = self._2_bn_proj(x)
        x = self._2_act_proj(x)

        r2 = x

        x1 = self._3_conv1(x)
        x1 = self._3_bn1(x1)

        x2 = self._3_conv1_h(x)
        x2 = self._3_bn1_h(x2)

        x3 = self._3_conv1_v(x)
        x3 = self._3_bn1_v(x3)

        x = x1+x2+x3
        x = self._3_act1(x)

        x1 = self._3_conv2(x)
        x1 = self._3_bn2(x1)

        x2 = self._3_conv2_h(x)
        x2 = self._3_bn2_h(x2)

        x3 = self._3_conv2_v(x)
        x3 = self._3_bn2_v(x3)
        
        x = x1+x2+x3

        x = x + r2
        x = self._3_act2(x)
        r3 = x

        x1 = self._4_conv1(x)
        x1 = self._4_bn1(x1)

        x2 = self._4_conv1_h(x)
        x2 = self._4_bn1_h(x2)

        x3 = self._4_conv1_v(x)
        x3 = self._4_bn1_v(x3)

        x = x1+x2+x3
        x = self._4_act1(x)

        x1 = self._4_conv2(x)
        x1 = self._4_bn2(x1)

        x2 = self._4_conv2_h(x)
        x2 = self._4_bn2_h(x2)

        x3 = self._4_conv2_v(x)
        x3 = self._4_bn2_v(x3)
        
        x = x1+x2+x3

        x = x + r3 
        x = self._4_act2(x)

        x = r2 + x
        #x = torch.cat([r1, r2, r3, x], 1)

        x = self._4_conv_proj(x)
        x = self._4_bn_proj(x)
        x = self._4_act_proj(x)

        return x


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]

def downsample_conv(
        in_channels, out_channels, kernel_size, stride=1, dilation=1, first_dilation=None, norm_layer=None):
    norm_layer = norm_layer or nn.BatchNorm2d
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    first_dilation = (first_dilation or dilation) if kernel_size > 1 else 1
    p = get_padding(kernel_size, stride, first_dilation)

    return nn.Sequential(*[
        nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=p, dilation=first_dilation, bias=False),
        norm_layer(out_channels)
    ])

class BottleNeckBlockV2(nn.Module):
    def __init__(self, inplanes, first_planes, outplanes, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,):
        super(BottleNeckBlockV2, self).__init__()

        self.conv_dw = nn.Conv2d(first_planes, first_planes, groups = first_planes, kernel_size=3, padding=1, bias=False)
        self.bn_dw = norm_layer(first_planes)

        self.conv_dw_h = nn.Conv2d(first_planes, first_planes, groups = first_planes, kernel_size=(1,3), padding=(0,1), bias=False)
        self.bn_dw_h = norm_layer(first_planes)

        self.conv_dw_v = nn.Conv2d(first_planes, first_planes, groups = first_planes, kernel_size=(3,1), padding=(1,0), bias=False)
        self.bn_dw_v = norm_layer(first_planes)

        self.act_dw = act_layer(inplace=True)

        self.conv_pwl = nn.Conv2d(first_planes, outplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_pwl = norm_layer(outplanes)
        
        self.conv_dwl = nn.Conv2d(outplanes, outplanes, groups = outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_dwl = norm_layer(outplanes)

        self.conv_dwl_h = nn.Conv2d(outplanes, outplanes, groups = outplanes, kernel_size=(1,3), padding=(0,1), bias=False)
        self.bn_dwl_h = norm_layer(outplanes)

        self.conv_dwl_v = nn.Conv2d(outplanes, outplanes, groups = outplanes, kernel_size=(3,1), padding=(1,0), bias=False)
        self.bn_dwl_v = norm_layer(outplanes)

    def forward(self, x):
        residual = x

        x1 = self.conv_dw(x)
        x1 = self.bn_dw(x1)

        x2 = self.conv_dw_h(x)
        x2 = self.bn_dw_h(x2)

        x3 = self.conv_dw_v(x)
        x3 = self.bn_dw_v(x3)

        x = x1+x2+x3
        x = self.act_dw(x)

        x = self.conv_pwl(x)
        x = self.bn_pwl(x)

        x1 = self.conv_dwl(x)
        x1 = self.bn_dwl(x1)

        x2 = self.conv_dwl_h(x)
        x2 = self.bn_dwl_h(x2)

        x3 = self.conv_dwl_v(x)
        x3 = self.bn_dwl_v(x3)

        x = x1+x2+x3

        x = x + residual

        return x

class ExpandBlock(nn.Module):
    def __init__(self, inplanes, first_planes, outplanes, stride=1, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,):
        super(ExpandBlock, self).__init__()

        self.conv_pw = nn.Conv2d(inplanes, first_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_pw = norm_layer(first_planes)
        self.act_pw = act_layer(inplace=True)

        self.conv_dw = nn.Conv2d(first_planes, first_planes, groups = first_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_dw = norm_layer(first_planes)

        self.conv_dw_h = nn.Conv2d(first_planes, first_planes, groups = first_planes, kernel_size=(1,3), stride=stride, padding=(0,1), bias=False)
        self.bn_dw_h = norm_layer(first_planes)

        self.conv_dw_v = nn.Conv2d(first_planes, first_planes, groups = first_planes, kernel_size=(3,1), stride=stride, padding=(1,0), bias=False)
        self.bn_dw_v = norm_layer(first_planes)

        self.act_dw = act_layer(inplace=True)

        self.conv_pwl = nn.Conv2d(first_planes, outplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_pwl = norm_layer(outplanes)
        
        self.conv_dwl = nn.Conv2d(outplanes, outplanes, groups = outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_dwl = norm_layer(outplanes)

        self.conv_dwl_h = nn.Conv2d(outplanes, outplanes, groups = outplanes, kernel_size=(1,3), padding=(0,1), bias=False)
        self.bn_dwl_h = norm_layer(outplanes)

        self.conv_dwl_v = nn.Conv2d(outplanes, outplanes, groups = outplanes, kernel_size=(3,1), padding=(1,0), bias=False)
        self.bn_dwl_v = norm_layer(outplanes)

    def forward(self, x):

        x = self.conv_pw(x)
        x = self.bn_pw(x)
        x = self.act_pw(x)

        x1 = self.conv_dw(x)
        x1 = self.bn_dw(x1)

        x2 = self.conv_dw_h(x)
        x2 = self.bn_dw_h(x2)

        x3 = self.conv_dw_v(x)
        x3 = self.bn_dw_v(x3)

        x = x1+x2+x3
        x = self.act_dw(x)

        x = self.conv_pwl(x)
        x = self.bn_pwl(x)

        x1 = self.conv_dwl(x)
        x1 = self.bn_dwl(x1)

        x2 = self.conv_dwl_h(x)
        x2 = self.bn_dwl_h(x2)

        x3 = self.conv_dwl_v(x)
        x3 = self.bn_dwl_v(x3)

        x = x1+x2+x3
        
        return x

# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64,
#                  reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d):
#         super(BasicBlock, self).__init__()

#         assert cardinality == 1, 'BasicBlock only supports cardinality of 1'
#         assert base_width == 64, 'BasicBlock doest not support changing base width'
#         first_planes = planes // reduce_first
#         outplanes = planes * self.expansion
#         first_dilation = first_dilation or dilation

#         self.conv1 = nn.Conv2d(
#             inplanes, first_planes, kernel_size=3, stride=stride, padding=first_dilation,
#             dilation=first_dilation, bias=False)
#         self.bn1 = norm_layer(first_planes)
#         self.act1 = act_layer(inplace=True)
#         self.conv2 = nn.Conv2d(
#             first_planes, outplanes, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
#         self.bn2 = norm_layer(outplanes)
#         self.act2 = act_layer(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#         self.dilation = dilation

#     def zero_init_last_bn(self):
#         nn.init.zeros_(self.bn2.weight)

#     def forward(self, x):
#         residual = x

#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.act1(x)

#         x = self.conv2(x)
#         x = self.bn2(x)
#         if self.downsample is not None:
#             residual = self.downsample(residual)
#         x += residual
#         x = self.act2(x)

#         return x

def get_padding(kernel_size, stride, dilation=1):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


@BACKBONES.register_module
class RegMobileNetV2(nn.Module):
    def __init__(self, pretrained ='/data1/centernet/dla2.pth', block=BasicBlock, in_chans=3,
                 act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d):

        super(RegMobileNetV2, self).__init__()
        self.pretrained = pretrained
        # self.base = DLAMain([1, 1, 1, 2, 2, 1], [16, 32, 64, 128, 256, 512],
        #             block=BasicBlock)
        self.conv1 = nn.Conv2d(in_chans, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv1_h = nn.Conv2d(in_chans, 16, kernel_size=(1,3), stride=2, padding=(0,1), bias=False)
        self.conv1_v = nn.Conv2d(in_chans, 16, kernel_size=(3,1), stride=2, padding=(1,0), bias=False)
        self.bn1 = norm_layer(16)
        self.bn1_h = norm_layer(16)
        self.bn1_v = norm_layer(16)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2_h = nn.Conv2d(16, 32, kernel_size=(1,3), stride=1, padding=(0,1), bias=False)
        self.conv2_v = nn.Conv2d(16, 32, kernel_size=(3,1), stride=1, padding=(1,0), bias=False)
        self.bn2 = norm_layer(32)
        self.bn2_h = norm_layer(32)
        self.bn2_v = norm_layer(32)
        self.act2 = act_layer(inplace=True)

        self.layer3 = DlaBlock1(32,48,48)
        self.layer4 = DlaBlock2(48,128,128)
        self.layer5 = DlaBlock2(128,288,288)
        self.layer6 = DlaBlock1(288,640,640)
        
        self.conv6 = nn.Conv2d(640,640, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn6 = norm_layer(640)
        self.act6 = act_layer(inplace=True)

        # self.conv1 = nn.Conv2d(in_chans, 32, kernel_size=3, stride=2, padding=1, bias=False)
        # self.conv1_h = nn.Conv2d(in_chans, 32, kernel_size=(1,3), stride=2, padding=(0,1), bias=False)
        # self.conv1_v = nn.Conv2d(in_chans, 32, kernel_size=(3,1), stride=2, padding=(1,0), bias=False)
        # self.bn1 = norm_layer(32)
        # self.bn1_h = norm_layer(32)
        # self.bn1_v = norm_layer(32)
        # self.act1 = act_layer(inplace=True)

        # self.layer2_1 = ExpandBlock(32,80,80,stride=2)
        # self.layer3_1 = ExpandBlock(80,240,240,stride=2)
        # self.layer3_2 = BottleNeckBlockV2(240,240,240)

        # self.layer4_1 = ExpandBlock(240,528,528,stride=2)
        # self.layer4_2 = BottleNeckBlockV2(528,528,528)
        # self.layer4_3 = BottleNeckBlockV2(528,528,528)
        # self.layer4_4 = BottleNeckBlockV2(528,528,528)

        # self.layer5_1 = ExpandBlock(528,1200,1200,stride=2)
        # self.layer5_2 = BottleNeckBlockV2(1200,1200,1200)

        
        # self.conv6 = nn.Conv2d(1200,1200, kernel_size=1, stride=1, padding=0, bias=False)
        # self.bn6 = norm_layer(1200)
        # self.act6 = act_layer(inplace=True)

        # self.l1 = ConvBnRelu(80, 64, 64)
        # self.l2 = ConvBnRelu(240, 128, 128)
        # self.l3 = ConvBnRelu(528, 256, 256)
        # self.l4 = ConvBnRelu(1200, 512, 512)

        self.level4 = ConvBnRelu(640, 288, 288)
        self.up4 = nn.ConvTranspose2d(288,288,4,stride=2,padding=1,output_padding=0,groups=288,bias=False)

        self.level3_1 = ConvBnRelu(288, 288, 288)
        self.level3_2 = ConvBnRelu(288, 48, 48)
        self.up3_1 = self.up3_1 = nn.ConvTranspose2d(48,48,8,stride=4,padding=2,output_padding=0,groups=48,bias=False)
        self.level3_3 = ConvBnRelu(288, 128, 128)
        self.up3_2 = nn.ConvTranspose2d(128,128,4,stride=2,padding=1,output_padding=0,groups=128,bias=False)
        self.level3_4 = ConvBnRelu(288, 128, 128)
        self.up3_3 = nn.ConvTranspose2d(128,128,4,stride=2,padding=1,output_padding=0,groups=128,bias=False)

        self.level2_1 = ConvBnRelu(128, 128, 128)
        self.level2_2 = ConvBnRelu(128, 128, 128)
        self.level2_3 = ConvBnRelu(128, 48, 48)
        self.level2_4 = ConvBnRelu(128, 48, 48)
        self.level2_5 = ConvBnRelu(128, 48, 48)
        self.level2_6 = ConvBnRelu(128, 48, 48)
        self.up2_1 = nn.ConvTranspose2d(48,48,4,stride=2,padding=1,output_padding=0,groups=48,bias=False)
        self.up2_2 = nn.ConvTranspose2d(48,48,4,stride=2,padding=1,output_padding=0,groups=48,bias=False)
        self.up2_3 = nn.ConvTranspose2d(48,48,4,stride=2,padding=1,output_padding=0,groups=48,bias=False)
        self.up2_4 = nn.ConvTranspose2d(48,48,4,stride=2,padding=1,output_padding=0,groups=48,bias=False)

        self.level1_1 = ConvBnRelu(48, 48, 48)
        self.level1_2 = ConvBnRelu(48, 48, 48)
        self.level1_3 = ConvBnRelu(48, 48, 48)
        self.level1_4 = ConvBnRelu(48, 48, 48)
        self.level1_5 = ConvBnRelu(48, 48, 48)
        # channels = [64, 128, 256, 512]
        # scales = [1,2,4,8]
        # self.dla_up = DLAUp(0, channels,scales)
        # self.ida_up = IDAUp(64, [64, 128, 256],[1,2,4])


        for n, m in self.named_modules():
            # if isinstance(m, nn.Conv2d):
            #     nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # elif isinstance(m, nn.BatchNorm2d):
            #     nn.init.constant_(m.weight, 1.)
            #     nn.init.constant_(m.bias, 0.)
            if isinstance(m, nn.ConvTranspose2d):
                fill_up_weights(m)

        # for m in [self.up4, self.up3_1, self.up3_2, self.up3_3, self.up2_1, self.up2_2, self.up2_3, self.up2_4]:
        #     m.eval()
        #     for param in m.parameters():
        #         param.requires_grad = False

        self.load_pretrained_model()

        #print(self.state_dict()['conv6.weight'])

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, reduce_first=1,
                    avg_down=False, down_kernel_size=1, **kwargs):
        downsample = None
        first_dilation = 1 if dilation in (1, 2) else 2
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample_args = dict(
                in_channels=self.inplanes, out_channels=planes * block.expansion, kernel_size=down_kernel_size,
                stride=stride, dilation=dilation, first_dilation=first_dilation, norm_layer=kwargs.get('norm_layer'))
            downsample = downsample_avg(**downsample_args) if avg_down else downsample_conv(**downsample_args)

        block_kwargs = dict(
            cardinality=self.cardinality, base_width=self.base_width, reduce_first=reduce_first,
            dilation=dilation, **kwargs)
        layers = [block(self.inplanes, planes, stride, downsample, first_dilation=first_dilation, **block_kwargs)]
        self.inplanes = planes * block.expansion
        layers += [block(self.inplanes, planes, **block_kwargs) for _ in range(1, blocks)]

        return nn.Sequential(*layers)

    def forward(self, x):

        # l = self.base(x)[2:]
        # # #l = [l1, l2, l3, x]
        # # x = self.dla_up(l)
        # # y = []
        # # for i in range(3):
        # #     y.append(x[i].clone())
        # # self.ida_up(y, 0, len(y))
        # # return y[-1]

        # # l = self.base(x)
        # l1 = l[0]
        # l2 = l[1]
        # l3 = l[2]
        # l4 = l[3]
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x2 = self.conv1_h(x)
        x2 = self.bn1_h(x2)
        x3 = self.conv1_v(x)
        x3 = self.bn1_v(x3)
        x = x1 + x2 + x3
        x = self.act1(x)

        x1 = self.conv2(x)
        x1 = self.bn2(x1)
        x2 = self.conv2_h(x)
        x2 = self.bn2_h(x2)
        x3 = self.conv2_v(x)
        x3 = self.bn2_v(x3)
        x = x1 + x2 + x3
        x = self.act2(x)

        x = self.layer3(x)
        l1 = x
        x = self.layer4(x)
        l2 = x
        x = self.layer5(x)
        l3 = x
        x = self.layer6(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.act6(x)
        l4 = x


        # x1 = self.conv1(x)
        # x1 = self.bn1(x1)
        # x2 = self.conv1_h(x)
        # x2 = self.bn1_h(x2)
        # x3 = self.conv1_v(x)
        # x3 = self.bn1_v(x3)
        # x = x1 + x2 + x3
        # x = self.act1(x)

        # x = self.layer2_1(x)
        # l1 = self.l1(x) 

        # x = self.layer3_1(x)
        # x = self.layer3_2(x)
        # l2 = self.l2(x) 
        # x = self.layer4_1(x)
        # x = self.layer4_2(x)
        # x = self.layer4_3(x)
        # x = self.layer4_4(x)
        # l3 = self.l3(x)
        # x = self.layer5_1(x)
        # x = self.layer5_2(x)
        # x = self.conv6(x)
        # x = self.bn6(x)
        # x = self.act6(x)
        # l4 = self.l4(x)


        l4 = self.level4(l4)
        l4 = self.up4(l4)
        l3_b = l3 + l4
        l3_b = self.level3_1(l3_b)
        l3_out = self.level3_2(l3_b)
        l3_out = self.up3_1(l3_out)

        l2_1 = self.level3_3(l3)
        l2_1 = self.up3_2(l2_1)
        l2_1 = l2 + l2_1
        l2_1 = self.level2_1(l2_1)

        l2_2 = self.level3_4(l3_b)
        l2_2 = self.up3_3(l2_2)

        l2_3 = self.level2_2(l2_1+l2_2)
        l2_out = self.level2_3(l2_3)
        l2_out = self.up2_1(l2_out)

        l1_2 = self.level2_4(l2)
        l1_2 = self.up2_2(l1_2)
        l1_2 = l1 + l1_2
        l1_2 = self.level1_1(l1_2) + self.up2_3(self.level2_5(l2_1))
        l1_2 = self.level1_2(l1_2) + self.up2_4(self.level2_6(l2_3))
        l1_out = self.level1_3(l1_2)
        l1_out = self.level1_4(l1_out + l2_out)
        out = self.level1_5(l1_out + l3_out)

        return out

    def load_pretrained_model(self):
        print(self.pretrained)
        state_dict = load_state_dict(self.pretrained)
        self.load_state_dict(state_dict, strict=False)

    def init_weights(self, pretrained=None):
        print('initializing weights')