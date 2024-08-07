import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base_model import BaseModel
from ..components import GagBlock, GagBasicBlock, SemanticPyramidPooling, PackSequenceWrapper, SeparateFCs, \
    SeparateBNNecks
from ..graph import Graph
import numpy as np


class GagFormer(BaseModel):

    def build_network(self, model_cfg):
        in_c = model_cfg['in_c']
        self.num_class = model_cfg['num_class']
        self.joint_format = model_cfg['joint_format']
        self.graph = Graph(joint_format=self.joint_format, max_hop=3)
        self.share_num = model_cfg['share_num']
        #### Network Define ####
        # ajaceny matrix
        self.A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        # data normalization

        ## joints_stream
        self.joints_stream = GagStream(in_c=in_c, first_c=4, share_num=self.share_num, is_Embed=True, graph=self.graph,joint_format=self.joint_format)
        ## bones_stream
        self.bones_stream = GagStream(in_c=in_c, first_c=2, share_num=self.share_num, is_Embed=True, graph=self.graph,joint_format=self.joint_format)
        # # # ## vel_stream
        self.vel_stream = GagStream(in_c=in_c, first_c=4, share_num=self.share_num, is_Embed=True, graph=self.graph,joint_format=self.joint_format)
        # # ## angle_stream
        # self.angle_stream = GagStream(in_c=in_c, first_c=2, share_num=self.share_num)

        # Head
        self.head = SeparateFCs(parts_num=12, in_channels=256, out_channels=256)
        # BNneck
        self.BNNecks = SeparateBNNecks(class_num=self.num_class, in_channels=256, parts_num=12)

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        x = ipts[0]
        pose = x
        N, C, T, V, M = x.size()
        if len(x.size()) == 4:
            x = x.unsqueeze(1)
        del ipts
        x_joints = x[:, :4, ...]
        x_bones = x[:, 4:6, ...]
        x_vel = x[:, 6:10, ...]
        # x_angle = x[:, 10:12, ...]
        # branch feature extractor
        fc_joints = self.joints_stream(x_joints, seqL)
        fc_bones = self.bones_stream(x_bones, seqL)
        fc_vel = self.vel_stream(x_vel, seqL)
        # fc_angle = self.angle_stream(x_angle, seqL)

        # feature fusion
        feature = torch.cat([fc_joints, fc_bones, fc_vel], -1)

        embed_1 = self.head(feature)
        embed_2, logits = self.BNNecks(embed_1)
        embed = embed_1
        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs}
            },
            'visual_summary': {
            },
            'inference_feat': {
                'embeddings': embed_2
            }
        }
        return retval


class GagStream(nn.Module):
    def __init__(self, in_c, first_c, share_num, is_Embed, graph, joint_format='coco'):
        super(GagStream, self).__init__()
        in_c = in_c
        first_c = first_c
        share_num = share_num
        self.graph = graph
        self.A = torch.from_numpy(self.graph.A.astype(np.float32))
        num_point = self.A.shape[-1]
        self.data_bn = nn.BatchNorm1d(first_c * num_point)
        self.joint_format = joint_format
        ## backbone
        ## main_stream
        main_stream = []
        # stage1
        main_stream.append(GagBasicBlock(in_channel=first_c, out_channel=in_c[1], A=self.A, is_Embed=is_Embed, num_point=num_point, joint_format=joint_format))
        # stage2
        for i in range(1, share_num):
            main_stream.append(GagBlock(in_channel=in_c[i], out_channel=in_c[i + 1], A=self.A, is_Embed=is_Embed, num_point=num_point, joint_format=joint_format))

        self.main_stream = nn.ModuleList(main_stream)
        parts2_1 = []
        for i in range(share_num, len(in_c) - 1):
            parts2_1.append(GagBlock(in_channel=in_c[i], out_channel=in_c[i + 1], A=self.A, is_Embed=is_Embed, num_point=num_point, joint_format=joint_format))
        self.parts2_1 = nn.ModuleList(parts2_1)

        ## parts2_2
        parts2_2 = []
        for i in range(share_num, len(in_c) - 1):
            parts2_2.append(GagBlock(in_channel=in_c[i], out_channel=in_c[i + 1], A=self.A, is_Embed=is_Embed, num_point=num_point, joint_format=joint_format))
        self.parts2_2 = nn.ModuleList(parts2_2)
        # ## parts3_1
        # parts3_1 = []
        # for i in range(share_num, len(in_c) - 1):
        #     parts3_1.append(GagBlock(in_channel=in_c[i], out_channel=in_c[i + 1], A=self.A, num_point=num_point))
        # self.parts3_1 = nn.ModuleList(parts3_1)
        # ## parts3_2
        # parts3_2 = []
        # for i in range(share_num, len(in_c) - 1):
        #     parts3_2.append(GagBlock(in_channel=in_c[i], out_channel=in_c[i + 1], A=self.A, num_point=num_point))
        # self.parts3_2 = nn.ModuleList(parts3_2)
        # ## parts4_1
        # parts4_1 = []
        # for i in range(share_num, len(in_c) - 1):
        #     parts4_1.append(GagBlock(in_channel=in_c[i], out_channel=in_c[i + 1], A=self.A, num_point=num_point))
        # self.parts4_1 = nn.ModuleList(parts4_1)
        # ## parts4_2
        # parts4_2 = []
        # for i in range(share_num, len(in_c) - 1):
        #     parts4_2.append(GagBlock(in_channel=in_c[i], out_channel=in_c[i + 1], A=self.A, num_point=num_point))
        # self.parts4_2 = nn.ModuleList(parts4_2)
        self.spp = SemanticPyramidPooling(joint_format=self.joint_format)
        self.set_pooling = PackSequenceWrapper(torch.max)

    def forward(self, x, seqL):
        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x_main = x
        for i, m in enumerate(self.main_stream):
            x_main = m(x_main, part=1)

        x_multi_brch = []
        ## parts2_1
        x_this_part = x_main.clone()
        for i, m in enumerate(self.parts2_1):
            x_this_part = m(x_this_part, part=2)

        x_this_part = self.set_pooling(x_this_part, seqL, options={"dim": 2})[0]
        # V Pooling
        temp = self.spp(x_this_part, part=2)
        x_multi_brch.append(temp)

        ## parts2_2
        x_this_part = x_main.clone()
        for i, m in enumerate(self.parts2_2):
            x_this_part = m(x_this_part, part=3)

        x_this_part = self.set_pooling(x_this_part, seqL, options={"dim": 2})[0]
        # V Pooling
        temp = self.spp(x_this_part, part=3)
        x_multi_brch.append(temp)

        # # 3-part-1 #
        # x_this_part = x_main.clone()
        # for i, m in enumerate(self.parts3_1):
        #     x_this_part = m(x_this_part, part=4)
        # # Set Pooling
        # x_this_part = self.set_pooling(x_this_part, seqL, options={"dim": 2})[0]
        # # V Pooling
        # temp = self.spp(x_this_part, part=4)
        # x_multi_brch.append(temp)

        # ## 3part-2 #
        # x_this_part = x_main.clone()
        # for i, m in enumerate(self.parts3_2):
        #     x_this_part = m(x_this_part, part=5)
        # # Set Pooling
        # x_this_part = self.set_pooling(x_this_part, seqL, options={"dim": 2})[0]
        # # V Pooling
        # temp = self.spp(x_this_part, part=5)
        # x_multi_brch.append(temp)

        ## 4part_1#
        # x_this_part = x_main.clone()
        # for i, m in enumerate(self.parts4_1):
        #     x_this_part = m(x_this_part, part=6)
        # x_this_part = self.set_pooling(x_this_part, seqL, options={"dim": 2})[0]
        # # V Pooling
        # temp = self.spp(x_this_part, part=6)
        # x_multi_brch.append(temp)
        # ## 4parts_2#
        # x_this_part = x_main.clone()
        # for i, m in enumerate(self.parts4_2):
        #     x_this_part = m(x_this_part, part=7)
        # x_this_part = self.set_pooling(x_this_part, seqL, options={"dim": 2})[0]
        # # V Pooling
        # temp = self.spp(x_this_part, part=7)
        # x_multi_brch.append(temp)

        x_p = torch.cat(x_multi_brch, -1)
        return x_p


