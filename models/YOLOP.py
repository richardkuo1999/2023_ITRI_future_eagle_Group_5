import os
import sys
import math
import logging
from pathlib import Path
from copy import deepcopy

import torch
import torch.nn as nn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.metrics import  SegmentationMetric
from utils.torch_utils import time_synchronized, initialize_weights, model_info,\
                            select_device
from utils.general import make_divisible
from torch.nn import Upsample
from models.common import *


logger = logging.getLogger(__name__)

class Model(nn.Module):
    def __init__(self, cfg, nc, anchors=None, ch=3, ):
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml
            with open(cfg, 'r', encoding='utf-8') as f:
                self.yaml = yaml.load(f, Loader=yaml.SafeLoader)  # load cfg

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        self.HeadOut_idx = self.yaml['HeadOut']
        if nc:
            logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value

        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        # self.model.HeadOut = self.model[self.HeadOut_idx]

        # Init weights, biases
        initialize_weights(self)
        self.info()
        logger.info('')

    def forward(self, x):
        cache = []
        # print(x.size())
        for i, block in enumerate(self.model):
            # print(i, block)
            if block.f != -1:
                x = cache[block.f] if isinstance(block.f, int) else [x if j == -1 else cache[j] for j in block.f]       #calculate concat detect
            x = block(x)

            if i == self.HeadOut_idx:
                m=nn.Sigmoid()
                out = m(x)
    
            cache.append(x if block.i in self.save else None)
        return out 
            

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

def parse_model(d, ch):
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    gd, gw = d['depth_multiple'], d['width_multiple']
    nc = d['nc']


    
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['Neck'] + d['Head']):
        # print(i,f, n, m, args)
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass
        
        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [nn.Conv2d, Conv, RepConv, SPP, SPPCSPC, Focus, 
                Bottleneck, BottleneckCSP]:
            c1, c2 = ch[f], args[0]
            # if c2 != no:  # if not output
            #     c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [SPPCSPC]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == "__main__":
    cfg = 'C:/Users/user/Desktop/ITRI_engle_project/cfg/YOLOv7_bT2.yaml'
    device = select_device('', batch_size=1)

    model = Model(cfg, 2).to(device)
    input = torch.randn((1, 3, 640, 640)).to(device, non_blocking=True)
    output = model(input)
    print('output:', output.size())
