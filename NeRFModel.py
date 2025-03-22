import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NeRFmodel(nn.Module):
    def __init__(self, embed_pos_L, embed_direction_L):
        super(NeRFmodel, self).__init__()
        #############################
        # network initialization

        D=8 
        W=256
        skips=[4]

        # First layer
        self.input_ch = embed_pos_L
        self.input_ch_dir = embed_direction_L

        # Positional encoding input layers
        self.pts_linears = nn.ModuleList(
            [nn.Linear(embed_pos_L, W)] + [nn.Linear(W, W) if i not in skips else nn.Linear(W + embed_pos_L, W) for i in range(D-1)]
        )

        # Density output (sigma)
        self.sigma_out = nn.Linear(W, 1)  

        # Feature output (for RGB computation)
        self.feature_out = nn.Linear(W, W)

        # View-dependent color computation
        self.dir_linear = nn.Linear(embed_direction_L + W, W // 2)
        self.rgb_out = nn.Linear(W // 2, 3)


        #############################



    def position_encoding(self, x, L):
        #############################
        # Implement position encoding 

        y = []

        for i in range(len(x)):

            values = []

            for j in range(L):

                exp = (2^j)*torch.pi*x[i]
                values.append(torch.sin(exp))
                values.append(torch.cos(exp))

            y.append(torch.tensor(values,dtype=torch.float32))


        return torch.stack(y)

        #############################



    def forward(self, pos, direction):

        #############################
        """
        x: (batch, input_ch) - Positional encoding of 3D location
        d: (batch, input_ch_dir) - Positional encoding of view direction
        """
        h = pos
        for i, layer in enumerate(self.pts_linears):
            h = layer(h)
            h = F.relu(h)
            if i in [4]:  # Skip connection
                h = torch.cat([h, pos], dim=-1)

        sigma = self.sigma_out(h)  # Density σ
        feature = self.feature_out(h)  # Intermediate feature vector

        # Concatenate viewing direction encoding with features
        h = torch.cat([feature, direction], dim=-1)
        h = self.dir_linear(h)
        h = F.relu(h)

        rgb = self.rgb_out(h)  # RGB color output

        return rgb, sigma

        #############################

