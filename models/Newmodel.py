import math
from collections import OrderedDict

import torch
import torch
from torch import nn
import torch.nn.functional as F

from models.common import Conv, E_ELAN, MPConv, OverlapPatchEmbed, MLP, resize

def check_anchor_order(m):
    # Check anchor order against stride order for YOLOv5 Detect() module m, and correct if necessary
    a = m.anchor_grid.prod(-1).view(-1)  # anchor area
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da.sign() != ds.sign():  # same order
        print('Reversing anchor order')
        m.anchors[:] = m.anchors.flip(0)
        m.anchor_grid[:] = m.anchor_grid.flip(0)

def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
        # elif t in [nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True

class UNext(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.layer_1 = nn.Sequential(OrderedDict([
                            ('conv1', Conv(3,32,3,1)),
                            ('conv2', Conv(32,64,3,2)),
                            ]))    
        self.layer_2 = nn.Sequential(OrderedDict([
                            ('conv1', Conv(64,64,3,1)),
                            ('conv2', Conv(64,128,3,2)),
                            ('E_ELAN', E_ELAN(128,256)),
                            ('conv3', Conv(256,256,1,1)),
                            ]))  
        self.layer_3 = nn.Sequential(OrderedDict([
                            ('MPConv', MPConv(256,256)),
                            ('E_ELAN', E_ELAN(256,512)),
                            ('conv', Conv(512,512,1,1)),
                            ]))  
        self.layer_4 = nn.Sequential(OrderedDict([
                            ('MPConv', MPConv(512,512)),
                            ('E_ELAN', E_ELAN(512,1024)),
                            ('conv', Conv(1024,1024,1,1)),
                            ]))  
                            
        # FIXME img_size embed_dims
        # self.patch_embed1 = OverlapPatchEmbed(img_size=640 // 16, patch_size=3, stride=2, in_chans=1024,
        #                                       embed_dim=1024)
        # self.patch_embed2 = OverlapPatchEmbed(img_size=640 // 8, patch_size=3, stride=2, in_chans=512,
        #                                       embed_dim=1024)
        # self.patch_embed3 = OverlapPatchEmbed(img_size=640 // 4, patch_size=3, stride=2, in_chans=256,
        #                                       embed_dim=1024)
        
        # self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        # FIXME embedding_dim = 1024
        self.linear_c4 = MLP(input_dim=1024, embed_dim=1024)
        self.linear_c3 = MLP(input_dim=512, embed_dim=1024)
        self.linear_c2 = MLP(input_dim=256, embed_dim=1024)
        self.linear_c1 = MLP(input_dim=64, embed_dim=1024)
        self.linear_c = MLP(input_dim=1024, embed_dim=1024)
        self.linear_conv1 = Conv(1024,1024,1,1)

        self.de4_conv = Conv(1024,1024,1,1)
        self.linear_c5 = MLP(input_dim=2048, embed_dim=512)
        
        self.de3_conv = Conv(512,512,1,1)
        self.linear_c6 = MLP(input_dim=1024, embed_dim=256)

        self.de2_conv = Conv(256,256,1,1)
        self.linear_c7 = MLP(input_dim=512, embed_dim=64)

        self.de1_conv = Conv(64,64,1,1)
        self.linear_c8 = MLP(input_dim=128, embed_dim=32)

        self.pred = Conv(32,nc,1,1)

    def forward(self, x):
        out = self.layer_1(x)
        c1 = out

        out = self.layer_2(out)
        c2 = out

        out = self.layer_3(out)
        c3 = out

        out = self.layer_4(out)
        c4 = out

        n, _, h, w = c4.shape
        
        # _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        # # _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        # _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        # _c3 = resize(_c3, size=c4.size()[2:],mode='bilinear',align_corners=False)

        # _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        # _c2 = resize(_c2, size=c4.size()[2:],mode='bilinear',align_corners=False)

        # _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])
        # _c1 = resize(_c1, size=c4.size()[2:],mode='bilinear',align_corners=False)

        # _c = torch.cat([_c4, _c3, _c2, _c1], dim=1)
        _c = self.linear_c(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c = resize(_c, size=c4.size()[2:],mode='bilinear',align_corners=False)

        
        c5 = torch.cat([self.linear_conv1(_c),self.de4_conv(c4)], dim=1)
        c5 = self.linear_c5(c5).permute(0,2,1).reshape(n, -1, c5.shape[2], c5.shape[3])
        c5 = resize(c5, size=c3.size()[2:],mode='bilinear',align_corners=False)
        
        c6 = torch.cat([c5,self.de3_conv(c3)], dim=1)
        c6 = self.linear_c6(c6).permute(0,2,1).reshape(n, -1, c6.shape[2], c6.shape[3])
        c6 = resize(c6, size=c2.size()[2:],mode='bilinear',align_corners=False)

        c7 = torch.cat([c6,self.de2_conv(c2)], dim=1)
        c7 = self.linear_c7(c7).permute(0,2,1).reshape(n, -1, c7.shape[2], c7.shape[3])
        c7 = resize(c7, size=c1.size()[2:],mode='bilinear',align_corners=False)

        c8 = torch.cat([c7,self.de1_conv(c1)], dim=1)
        c8 = self.linear_c8(c8).permute(0,2,1).reshape(n, -1, c8.shape[2], c8.shape[3])
        c8 = resize(c8, size=x.size()[2:],mode='bilinear',align_corners=False)
        # emb1,H1,W1 = self.patch_embed1(s4) # 1  400 1024, 20 20
        # emb2,H2,W2 = self.patch_embed2(s3) # 1 1600 1024, 40 40
        # emb3,H3,W3 = self.patch_embed3(s2) # 1 6400 1024, 80 80
        
        return self.pred(c8)
    


class Model(nn.Module):
  def __init__(self, cfg = None, nc = 2, ch=3, ):
    super(Model, self).__init__()
    self.model = UNext(nc)

  def forward(self, x):         
    return self.model(x)



if __name__ == '__main__':
  model = Model(nc=2).cuda()
  input = torch.ones(1,3,640,640).cuda()
  print(model)
#   model.eval()
  output = model(input)
  print(output.shape)
#   print(output[0][0].shape)
#   print(output[1].shape)
#   print(output[2].shape)
