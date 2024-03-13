import torch
import torch.nn as nn
import torch.nn.functional as F
from src.smb.level import MarioLevel
from performer_pytorch import SelfAttention

class CustomUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, target_size):
        super(CustomUpsample, self).__init__()

        if target_size == 4:  # Upsampling from 2x2 to 4x4
            stride, kernel_size, padding = 1, 3, 0
        elif target_size == 7:  # Upsampling from 4x4 to 7x7
            stride, kernel_size, padding = 2, 3, 1
        elif target_size == 8:  # Upsampling from 4x4 to 8x8
            stride, kernel_size, padding = 2, 4, 1
        elif target_size == 14:  # Upsampling from 7x7 to 14x14
            stride, kernel_size, padding = 2, 2, 0
        elif target_size == 16:  # Upsampling from 8x8 to 16x16
            stride, kernel_size, padding = 2, 4, 1
        else:
            raise ValueError("Invalid target_size specified.")

        self.upsample = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

    def forward(self, x):
        return self.upsample(x)

class PerformerSelfAttention(nn.Module):
    def __init__(self, channels, size, n_heads=4):
        super(PerformerSelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.n_heads = n_heads

        # Use PerformerSelfAttention from performer-pytorch
        self.performer_attention = SelfAttention(
            dim=channels,
            heads=self.n_heads
        )

        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)

        # Adapt the input shape for PerformerSelfAttention
        query_key_value = x_ln.view(batch_size, self.size * self.size, self.channels)
        attention_value = self.performer_attention(query_key_value)

        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels

        # Define the two convolution layers
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.group_norm1 = nn.GroupNorm(1, mid_channels)
        self.gelu1 = nn.GELU()
        
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.group_norm2 = nn.GroupNorm(1, out_channels)

    def forward(self, x):
        # Apply the first convolution layer
        x1 = self.conv1(x)
        x1 = self.group_norm1(x1)
        x1 = self.gelu1(x1)

        # Apply the second convolution layer
        x2 = self.conv2(x1)
        x2 = self.group_norm2(x2)

        # Apply residual connection and GELU activation
        if self.residual:
            return F.gelu(x + x2)
        else:
            return x2

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=32):
        super().__init__()
        # Max pooling followed by two DoubleConv layers
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2, ceil_mode=True),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        # Embedding layer to incorporate time information
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        # Apply the embedding layer and broadcast the output to match spatial dimensions
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=32, target_size=7):
        super().__init__()

        self.up = CustomUpsample(in_channels=int(in_channels/2), out_channels=int(in_channels/2), target_size=target_size)

        # DoubleConv layers after concatenation
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        # Embedding layer to incorporate time information
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        # Upsample the input tensor
        x = self.up(x)
        # Concatenate the upsampled tensor with the skip tensor from the encoder

        # print('x.shape: {}, skip_x.shape: {}'.format(x.shape, skip_x.shape))

        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        # Apply the embedding layer and broadcast the output to match spatial dimensions
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

class UNet(nn.Module):
    def __init__(self, c_in=MarioLevel.n_types, c_out=MarioLevel.n_types, time_dim=32, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim

        self.inc = DoubleConv(c_in, 64)     # 64x16x16
        self.down1 = Down(64, 128)          # 128x8x8
        self.sa1 = PerformerSelfAttention(128, 8)    # 128x8x8
        self.down2 = Down(128, 256)         # 256x4x4
        self.sa2 = PerformerSelfAttention(256, 4)    # 256x4x4
        self.down3 = Down(256, 256)         # 256x2x2
        self.sa3 = PerformerSelfAttention(256, 2)    # 256x2x2

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)    # 256x2x2

        self.up1 = Up(512, 128, target_size=4)       # 128x4x4
        self.sa4 = PerformerSelfAttention(128, 4)    # 256x4x4
        self.up2 = Up(256, 64, target_size=8)       # 64x8x8
        self.sa5 = PerformerSelfAttention(64, 8)    # 128x8x8
        self.up3 = Up(128, 64, target_size=16)       # 64x16x16
        self.sa6 = PerformerSelfAttention(64, 16)    # 64x16x16
        self.outc = nn.Conv2d(64, c_out, kernel_size=1) # 11x16x16

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x1 = self.inc(x)            # 64x16x16
        x2 = self.down1(x1, t)      # 128x8x8
        # try:
        x2 = self.sa1(x2)           # 128x8x8
        # except RuntimeError:
        #     print(x.shape, x2.shape)
        x3 = self.down2(x2, t)      # 256x4x4
        x3 = self.sa2(x3)           # 256x4x4
        x4 = self.down3(x3, t)      # 256x2x2
        x4 = self.sa3(x4)           # 256x2x2

        x4 = self.bot1(x4)          
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)          # 256x2x2

        x = self.up1(x4, x3, t)     # 256x4x4
        x = self.sa4(x)             # 256x4x4
        x = self.up2(x, x2, t)      # 128x8x8
        x = self.sa5(x)             # 128x8x8
        x = self.up3(x, x1, t)      # 64x16x16
        x = self.sa6(x)             # 64x16x16
        output = self.outc(x)       # 11x16x16
        return output

