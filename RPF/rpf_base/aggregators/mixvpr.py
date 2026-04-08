import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class FeatureMixerLayerMean(nn.Module):
    def __init__(self, in_dim, mlp_ratio=1):
        super().__init__()
        self.mix = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, int(in_dim * mlp_ratio)),
            nn.ReLU(),
            nn.Linear(int(in_dim * mlp_ratio), in_dim),
        )

    def forward(self, x):
        return x + self.mix(x)

class FeatureMixerLayerStd(nn.Module):
    def __init__(self, in_dim, mlp_ratio=1):
        super().__init__()
        self.mix = nn.Sequential(
            nn.Linear(in_dim, int(in_dim * mlp_ratio)),
            nn.ReLU(),
            nn.Linear(int(in_dim * mlp_ratio), in_dim),
        )
    def forward(self, x):
        return x + self.mix(x)

class VarianceMixVPR(nn.Module):
    def __init__(self,
                 in_channels=1024,
                 in_h=20,
                 in_w=20,
                 out_channels=512,
                 mix_depth=4,
                 mlp_ratio=1,
                 out_rows=4,
                 ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_rows = out_rows
        self.mix_depth = mix_depth
        self.mlp_ratio = mlp_ratio

        hw = in_h * in_w
        self.mix = nn.Sequential(*[FeatureMixerLayerStd(in_dim=hw, mlp_ratio=mlp_ratio) for _ in range(self.mix_depth)
                                   ])
        self.channel_proj = nn.Linear(in_channels, out_channels)
        self.row_proj = nn.Linear(hw, out_rows)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.flatten(2)  # [B, C, H*W]

        # Feature Mixing without normalization
        x = self.mix(x)  # [B, C, H*W]

        x = x.permute(0, 2, 1)  # [B, H*W, C]
        x = self.channel_proj(x)  # [B, H*W, out_c]
        x = x.permute(0, 2, 1)  # [B, out_c, H*W]
        x = self.row_proj(x)  # [B, out_c, out_rows]

        x = x.flatten(1)  # [B, D]
        raw_log_var=x
        var = F.softplus(raw_log_var )
        var = torch.clamp(var, min=1e-4)
        return var
class MixVPR(nn.Module):
    def __init__(self,
                 in_channels=1024,
                 flatten_dim=1536,
                 in_h=20,
                 in_w=20,
                 out_channels=512,
                 mix_depth=4,
                 mlp_ratio=1,
                 out_rows=4,
                 ) -> None:
        super().__init__()
        # self.in_h = in_h
        # self.in_w = in_w
        self.in_channels = in_channels  # depth of input feature maps

        self.out_channels = out_channels  # depth wise projection dimension
        self.out_rows = out_rows  # row wise projection dimesion

        self.mix_depth = mix_depth  # L the number of stacked FeatureMixers
        self.mlp_ratio = mlp_ratio  # ratio of the mid projection layer in the mixer block

        hw = in_h * in_w
        # hw = flatten_dim
        self.mix = nn.Sequential(*[FeatureMixerLayerMean(in_dim=hw, mlp_ratio=mlp_ratio) for _ in range(self.mix_depth)
        ])
        self.channel_proj = nn.Linear(in_channels, out_channels)
        self.row_proj = nn.Linear(hw, out_rows)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def forward(self, x,until=None):
        x = x.flatten(2)
        x = self.mix(x)  # Feature-Mixer
        x = x.permute(0, 2, 1)
        x = self.channel_proj(x)
        x = x.permute(0, 2, 1)
        if until is not None:
            return x
        x = self.row_proj(x)
        x = x.flatten(1)
        x1 = F.normalize(x, p=2, dim=-1)
        return x1


# -------------------------------------------------------------------------------
def print_nb_params(m):
    model_parameters = filter(lambda p: p.requires_grad, m.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Trainable parameters: {params / 1e6:.3}M')


def main():
    x = torch.randn(1, 1024, 20, 20).to(torch.device('cuda'))
    agg = MixVPR(
        in_channels=1024,
        in_h=20,
        in_w=20,
        # flatten_dim=1536,
        out_channels=1024,
        mix_depth=4,
        mlp_ratio=1,
        out_rows=4).to(torch.device('cuda'))

    # print_nb_params(agg)
    output = agg(x)
    # summary(agg, input_size=(320, 7, 7), batch_size=1, device='cuda')
    print(output.shape)


if __name__ == '__main__':
    main()
    # convert_to_onnx(r"E:\Pytorch_code\MixVPR\version_1\checkpoints\resnet50_epoch(34)_step(21910)_R1[0.9360]_R5[0.9821].ckpt")
