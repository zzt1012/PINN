
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# add configs.py path
file_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(file_path.split('cnn')[0]))
sys.path.append(os.path.join(file_path.split('Models')[0]))


activation_dict = \
    {'gelu': nn.GELU(), 'silu': nn.SiLU(), 'relu': nn.ReLU(), 'tanh': nn.Tanh(), 'leakyrelu': nn.LeakyReLU(),
     None: nn.SiLU()}

additional_attr = ['normalizer', 'raw_laplacian', 'return_latent',
                   'residual_type', 'norm_type', 'norm_eps', 'boundary_condition',
                   'upscaler_size', 'downscaler_size', 'spacial_dim', 'spacial_fc',
                   'regressor_activation', 'attn_activation',
                   'downscaler_activation', 'upscaler_activation',
                   'encoder_dropout', 'decoder_dropout', 'ffn_dropout']

class FcnSingle(nn.Module):
    def __init__(self, planes: list or tuple, activation="gelu", last_activation=False):
        # =============================================================================
        #     Inspired by M. Raissi a, P. Perdikaris b,∗, G.E. Karniadakis.
        #     "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems
        #     involving nonlinear partial differential equations".
        #     Journal of Computational Physics.
        # =============================================================================
        super(FcnSingle, self).__init__()
        self.planes = planes
        self.active = activation_dict[activation]

        self.layers = nn.ModuleList()
        for i in range(len(self.planes) - 2):
            self.layers.append(nn.Linear(self.planes[i], self.planes[i + 1]))
            self.layers.append(self.active)
        self.layers.append(nn.Linear(self.planes[-2], self.planes[-1]))

        if last_activation:
            self.layers.append(self.active)
        self.layers = nn.Sequential(*self.layers)  # *的作用是解包

        # self.reset_parameters()

    def reset_parameters(self):
        """
        weight initialize
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # nn.init.xavier_normal_(m.weight, gain=1)
                nn.init.xavier_uniform_(m.weight, gain=1)
                m.bias.data.zero_()

    def forward(self, in_var):
        """
        forward compute
        :param in_var: (batch_size, ..., input_dim)
        """
        out_var = self.layers(in_var)
        return out_var

class DeepONetMulti(nn.Module):
    # =============================================================================
    #     Inspired by L. Lu, J. Pengzhan, G.E. Karniadakis.
    #     "DeepONet: Learning nonlinear operators for identifying differential equations based on
    #     the universal approximation theorem of operators".
    #     arXiv:1910.03193v3 [cs.LG] 15 Apr 2020.
    # =============================================================================
    def __init__(self, input_dim: int, operator_dims: list, output_dim: int,
                 planes_branch: list, planes_trunk: list, activation='gelu'):
        """
        :param input_dim: int, the coordinates dim for trunk net
        :param operator_dims: list，the operate dims list for each branch net
        :param output_dim: int, the predicted variable dims
        :param planes_branch: list, the hidden layers dims for branch net
        :param planes_trunk: list, the hidden layers dims for trunk net
        :param operator_dims: list，the operate dims list for each branch net
        :param activation: activation function
        """
        super(DeepONetMulti, self).__init__()

        self.branches = nn.ModuleList() # 分支网络
        self.trunks = nn.ModuleList() # 主干网络
        for dim in operator_dims:
            self.branches.append(FcnSingle([dim] + planes_branch, activation=activation))# FcnSingle是从basic_layers里导入的
        for _ in range(output_dim):
            self.trunks.append(FcnSingle([input_dim] + planes_trunk, activation=activation))

        self.reset_parameters()

    def reset_parameters(self): # 初始化所有网络的参数
        """
        weight initialize
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # nn.init.xavier_normal_(m.weight, gain=1)
                nn.init.xavier_uniform_(m.weight, gain=1)
                m.bias.data.zero_()

    def forward(self, u_vars, y_var, size_set=True):
        """
        forward compute
        :param u_vars: tensor list[(batch_size, ..., operator_dims[0]), (batch_size, ..., operator_dims[1]), ...]
        :param y_var: (batch_size, ..., input_dim)
        :param size_set: bool, true for standard inputs, false for reduce points number in operator inputs
        """
        B = 1.
        for u_var, branch in zip(u_vars, self.branches):
            B *= branch(u_var)
        if not size_set:
            B_size = list(y_var.shape[1:-1])
            for i in range(len(B_size)):
                B = B.unsqueeze(1)
            B = torch.tile(B, [1, ] + B_size + [1, ])

        out_var = []
        for trunk in self.trunks:
            T = trunk(y_var)
            out_var.append(torch.sum(B * T, dim=-1)) # 用这种方式实现两个网络的乘积
        out_var = torch.stack(out_var, dim=-1)
        return out_var


if __name__ == "__main__":
    us = [torch.ones([10, 64, 256 * 2]), torch.ones([10, 64, 1])]
    x = torch.ones([10, 64, 2])
    layer = DeepONetMulti(input_dim=2, operator_dims=[256 * 2, 1], output_dim=5,
                          planes_branch=[64] * 3, planes_trunk=[64] * 2)
    y = layer(us, x)
    print(y.shape)