import math
import torch.nn as nn
from ..other_modules import Mish




class UnitConv2D(nn.Module):
    '''
    This class is used in GaitTR[TCN_ST] block.
    '''

    def __init__(self, D_in, D_out,  kernel_size=9, stride=1, dropout=0.5, bias=True):
        super(UnitConv2D,self).__init__()
        pad = int((kernel_size-1)/2) * 1
        self.conv = nn.Conv2d(D_in, D_out, kernel_size=(kernel_size, 1), padding=(pad, 0), stride=(stride, 1), bias=bias)
        self.bn = nn.BatchNorm2d(D_out)
        self.act = Mish()
        # self.dropout = nn.Dropout(dropout, inplace=False)
        #initalize
        self.conv_init(self.conv)

    def forward(self, x):
        # x = self.dropout(x)
        x = self.bn(self.act(self.conv(x)))
        return x

    def conv_init(self,module):
        n = module.out_channels
        for k in module.kernel_size:
            n = n*k
        module.weight.data.normal_(0, math.sqrt(2. / n))

class UnitDWConv(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,):
        super(UnitDWConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels, stride=stride)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))


    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
