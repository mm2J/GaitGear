import torch
import torch.nn as nn
import torch.nn.functional as F
from ...graph import GraphPartition
from ..other_modules import Mish


class GagAttention(nn.Module):
    def __init__(self, in_channels, out_channels, joint_format, incidence, is_Embed, num_point):
        super(GagAttention, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_point = num_point
        self.incidence = incidence
        self.is_Embed = is_Embed
        self.bn = nn.BatchNorm2d(self.out_channels)
        self.data_bn = nn.BatchNorm1d(self.in_channels * self.num_point)
        self.attention_conv = spatial_attention(in_channels=in_channels, out_channel=out_channels,
                                                joint_format=joint_format, A=incidence, is_Embed=is_Embed, num_point=self.num_point)
        self.act = Mish()

    def forward(self, x, part=None):
        N, C, T, V = x.size()
        # data normlization
        x = x.permute(0, 1, 3, 2).reshape(N, C * V, T)
        x = self.data_bn(x)
        x = x.reshape(N, C, V, T).permute(0, 1, 3, 2)
        # N, T, C, V > NT, C, 1, V
        xa = x.permute(0, 2, 1, 3).reshape(-1, C, 1, V)
        # spatial attention
        attn_out = self.attention_conv(xa, part)
        # N, T, C, V > N, C, T, V
        attn_out = attn_out.reshape(N, T, -1, V).permute(0, 2, 1, 3)
        y = self.bn(attn_out)
        y = self.act(y)
        return y


class spatial_attention(nn.Module):
    def __init__(self, in_channels, out_channel, num_point, joint_format, A, is_Embed, dk_factor=0.25, kernel_size=1, Nh=8, num=4,
                 stride=1):
        super(spatial_attention, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.A = A[0] + A[1] + A[2]
        self.dk = int(dk_factor * out_channel)
        self.dv = int(out_channel)
        self.num = num
        self.Nh = Nh
        self.num_point = num_point
        self.stride = stride
        self.is_Embed = is_Embed
        self.padding = (self.kernel_size - 1) // 2
        assert self.Nh != 0, "integer division or modulo by zero, Nh >= 1"
        assert self.dk % self.Nh == 0, "dk should be divided by Nh. (example: out_channels: 20, dk: 40, Nh: 4)"
        assert self.dv % self.Nh == 0, "dv should be divided by Nh. (example: out_channels: 20, dv: 4, Nh: 4)"
        assert stride in [1, 2], str(stride) + " Up to 2 strides are allowed."
        self.partition = GraphPartition(joint_format=joint_format)
        self.qkv_conv = nn.Conv2d(self.in_channels, 2 * self.dk + self.dv, kernel_size=self.kernel_size, stride=stride,
                                  padding=self.padding)

        self.edge_embed = nn.Parameter(torch.zeros_like(self.A))
        self.PA = nn.Parameter(torch.zeros_like(self.A))
        self.sigma = nn.Parameter(torch.zeros(1))

    def Partition2Mask(self, partition, num_point):
        part_mask = torch.zeros([num_point, num_point], dtype=torch.float)
        for part in partition:
            for i in part:
                for j in part:
                    part_mask[i, j] = 1.
        return part_mask

    def forward(self, x, part):
        # Input x
        # (batch_size, channels, 1, joints)
        B, _, T, V = x.size()
        self.A = self.A.cuda(x.get_device())


        flat_q, flat_k, flat_v, q, k, v = self.compute_flat_qkv(x, self.dk, self.dv, self.Nh)
        logits = torch.matmul(flat_q.transpose(2, 3), flat_k)

        if self.is_Embed:
            if part:
                partition = self.partition(part)
                num_point = self.num_point
                part_mask = self.Partition2Mask(partition, num_point)
                part_mask = part_mask.cuda(x.get_device())
                embed_weight = self.edge_embed * self.A @ part_mask
            else:
                embed_weight = self.edge_embed * self.A

            edge_embed = embed_weight * self.sigma
            edge_vector = edge_embed.clone()
            logits = logits + edge_embed + self.PA
            #logits = logits + self.PA
            weights = F.softmax(logits, dim=-1)
            attn_out = torch.matmul(weights, flat_v.transpose(2, 3))
            attn_out = torch.reshape(attn_out, (B, self.Nh, T, V, self.dv // self.Nh))
            attn_out = attn_out.permute(0, 1, 4, 2, 3)
            attn_out = self.combine_heads(attn_out)
            attn_out = attn_out + edge_vector.mean(dim=1)
        else:
            weights = F.softmax(logits, dim=-1)
            attn_out = torch.matmul(weights, flat_v.transpose(2, 3))
            attn_out = torch.reshape(attn_out, (B, self.Nh, T, V, self.dv // self.Nh))
            attn_out = attn_out.permute(0, 1, 4, 2, 3)
            attn_out = self.combine_heads(attn_out)
        return attn_out

    def compute_flat_qkv(self, x, dk, dv, Nh):
        qkv = self.qkv_conv(x)

        # T=1 in this case, because we are considering each frame separately
        N, _, T, V = qkv.size()

        q, k, v = torch.split(qkv, [dk, dk, dv], dim=1)
        q = self.split_heads(q, Nh)
        k = self.split_heads(k, Nh)
        v = self.split_heads(v, Nh)
        dkh = dk // Nh
        q = q * (dkh ** -0.5)
        flat_q = torch.reshape(q, (N, Nh, dkh, T * V))
        flat_k = torch.reshape(k, (N, Nh, dkh, T * V))
        flat_v = torch.reshape(v, (N, Nh, dv // self.Nh, T * V))
        return flat_q, flat_k, flat_v, q, k, v

    def split_heads(self, x, Nh):
        B, channels, T, V = x.size()
        ret_shape = (B, Nh, channels // Nh, T, V)
        split = torch.reshape(x, ret_shape)
        return split

    def combine_heads(self, x):
        batch, Nh, dv, T, V = x.size()
        ret_shape = (batch, Nh * dv, T, V)
        return torch.reshape(x, ret_shape)
