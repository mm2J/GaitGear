import torch.nn as nn
import torch
from .unit_conv2d import UnitDWConv

class TNN(nn.Module):
    def __init__(self, in_c, out_c, num=4):
        super(TNN, self).__init__()
        assert num % 2 == 0
        self.phase_oc = out_c // num
        kernel_size = [11, 11, 11, 11]
        padding = [int(kernel_size[i] - 1) // 2 for i in range(len(kernel_size))]
        self.conv1_1 = UnitDWConv(in_channels=in_c, out_channels=self.phase_oc,
                                 kernel_size=(kernel_size[0], 1), padding=(padding[0], 0))

        self.conv1_2 = UnitDWConv(in_channels=in_c, out_channels=self.phase_oc,
                                 kernel_size=(kernel_size[1], 1), padding=(padding[1], 0))

        self.conv1_3 = UnitDWConv(in_channels=in_c, out_channels=self.phase_oc,
                                 kernel_size=(kernel_size[2], 1), padding=(padding[2], 0))

        self.conv1_4 = UnitDWConv(in_channels=in_c, out_channels=self.phase_oc,
                                 kernel_size=(kernel_size[3], 1), padding=(padding[3], 0))

        self.act = nn.GELU()

        self.fusion1 = nn.Sequential(
            nn.Conv2d(in_channels=self.phase_oc*2, out_channels=self.phase_oc*2, kernel_size=1),
            nn.BatchNorm2d(self.phase_oc*2),
            nn.Sigmoid()
        )

        self.fusion2 = nn.Softmax(dim=2)

    def forward(self, x):
        x1 = self.act(self.conv1_1(x))
        x2 = self.act(self.conv1_2(x))
        x3 = self.act(self.conv1_3(x))
        x4 = self.act(self.conv1_4(x))
        fc_1 = torch.cat([x1, x2], dim=1)
        fc_2 = torch.cat([x3, x4], dim=1)
        fc_1 = self.fusion1(fc_1)
        fc_2 = self.fusion2(fc_2)
        res = torch.cat([fc_1, fc_2], dim=1)
        return res