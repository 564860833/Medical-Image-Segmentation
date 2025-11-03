import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.layers.adaptive_avgmax_pool import SelectAdaptivePool2d


# ---------------------------------------------------------------------------
# --- 步骤 1: 从 MSHFFA (common.py) 复制的核心代码 ---
# ---------------------------------------------------------------------------

def default_conv(in_channels, out_channels, kernel_size, bias=True, dilation=1):
    """
    MSHFFA 依赖: 默认卷积
    来源: resonwang/mshffa/.../Dwt_models/common.py
    """
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2) + dilation - 1, bias=bias, dilation=dilation)


def dwt_init(x):
    """
    MSHFFA 依赖: DWT 初始化函数
    来源: resonwang/mshffa/.../Dwt_models/common.py

    x = (B, C, H, W)
    x1:偶数行偶数列 x2:奇数行偶数列 x3:偶数行奇数列 x4: 奇数行奇数列
    返回：# (B, 4C, H/2, W/2) 4C:C(LL)-C(HL)-C(LH)-C(HH)
    """
    x01 = x[:, :, 0::2, :] / 2  # (B, C, H, W)-->(B, C, H/2, W)  取偶数行
    x02 = x[:, :, 1::2, :] / 2  # (B, C, H, W)-->(B, C, H/2, W)  取奇数行
    x1 = x01[:, :, :, 0::2]  # (B, C, H/2, W)-->(B, C, H/2, W/2) 取偶数列
    x2 = x02[:, :, :, 0::2]  # (B, C, H/2, W)-->(B, C, H/2, W/2) 取奇数列
    x3 = x01[:, :, :, 1::2]  # (B, C, H/2, W)-->(B, C, H/2, W/2) 取奇数列
    x4 = x02[:, :, :, 1::2]  # (B, C, H/2, W)-->(B, C, H/2, W/2) 取偶数列
    x_LL = x1 + x2 + x3 + x4  # (B, C, H/2, W/2)
    x_HL = -x1 - x2 + x3 + x4  # (B, C, H/2, W/2)
    x_LH = -x1 + x2 - x3 + x4  # (B, C, H/2, W/2)
    x_HH = x1 - x2 - x3 + x4  # (B, C, H/2, W/2)

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)  # (B, 4C, H/2, W/2)


class DWT(nn.Module):
    """
    MSHFFA 依赖: DWT 封装类
    来源: resonwang/mshffa/.../Dwt_models/common.py
    """

    def __init__(self, requires_grad=False):
        super(DWT, self).__init__()
        self.requires_grad = requires_grad

    def forward(self, x):
        return dwt_init(x)


class BBlock(nn.Module):
    """
    MSHFFA 依赖: 基础块 (用于 DWT_spatial_attention3)
    Conv-bn-act
    来源: resonwang/mshffa/.../Dwt_models/common.py
    """

    def __init__(
            self, conv, in_channels, out_channels, kernel_size, bias=True, bn=False, act=None):
        super(BBlock, self).__init__()
        m = []
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        self.body = nn.Sequential(*m)

    def forward(self, x):
        x = self.body(x)
        return x


class BBlock1(nn.Module):
    """
    MSHFFA 依赖: 基础块 (用于 DWT_spatial_attention3)
    Conv-bn-act
    来源: resonwang/mshffa/.../Dwt_models/common.py
    """

    def __init__(
            self, conv, in_channels, out_channels, kernel_size, bias=True, bn=False):
        super(BBlock1, self).__init__()
        m = []
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        # m.append(act)
        self.body = nn.Sequential(*m)

    def forward(self, x):
        x = self.body(x)
        return x


class BBlock_LowF(nn.Module):
    """
    MSHFFA 依赖: 低频处理块 (用于 DWT_spatial_attention3)
    Conv-bn-act
    来源: resonwang/mshffa/.../Dwt_models/common.py
    """

    def __init__(
            self, conv, in_channels, out_channels, kernel_size, bias=True, bn=False, act=nn.ReLU(True)):
        super(BBlock_LowF, self).__init__()
        m = []
        m.append(conv(in_channels, out_channels // 4, kernel_size=kernel_size, stride=2, padding=1,
                      bias=bias))  # 一次下采样  112-56
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        m.append(conv(out_channels // 4, out_channels // 2, kernel_size=kernel_size, stride=2, padding=1,
                      bias=bias))  # 二次下采样  56-28
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        m.append(conv(out_channels // 2, out_channels, kernel_size=kernel_size, stride=2, padding=1,
                      bias=bias))  # 三次下采样  28-14   (BS, 64, 14, 14)
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)

        m.append(SelectAdaptivePool2d(pool_type='avg'))  # (BS, 64, 14, 14)-->(BS, 64, 1, 1)
        m.append(nn.Flatten(1))  # (BS, 64)
        self.body = nn.Sequential(*m)

    def forward(self, x):
        x = self.body(x)
        return x


class DWT_spatial_attention3(nn.Module):
    """
    MSHFFA 核心模块
    来源: resonwang/mshffa/.../Dwt_models/common.py
    """

    def __init__(self, hidden_d=64):
        super().__init__()
        act = nn.ReLU(True)
        self.dwt = DWT()
        self.conv_l1 = BBlock_LowF(nn.Conv2d, 1, hidden_d, kernel_size=3, act=act)
        self.conv_h1 = BBlock(default_conv, 3, hidden_d, kernel_size=3, act=act)
        self.conv_l2 = BBlock_LowF(nn.Conv2d, 1, hidden_d, kernel_size=3, act=act)
        self.conv_h2 = BBlock(default_conv, 3, hidden_d, kernel_size=3, act=act)
        self.conv_l3 = BBlock_LowF(nn.Conv2d, 1, hidden_d, kernel_size=3, act=act)
        self.conv_h3 = BBlock(default_conv, 3, hidden_d, kernel_size=3, act=act)
        self.conv_sa1 = BBlock1(default_conv, 1, 1, kernel_size=3)
        self.conv_sa2 = BBlock1(default_conv, 1, 1, kernel_size=3)
        self.conv_sa3 = BBlock1(default_conv, 1, 1, kernel_size=3)
        self.sigmoid = nn.Sigmoid()
        self.norm_flag = 0
        self.norm = nn.LayerNorm(hidden_d)

    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        """
        ### Calculate scores between queries and keys
        This method can be overridden for other variations like relative attention.
        """
        # Calculate $Q K^\top$
        return torch.einsum('bihd,bjhd->bijh', query, key)  # query,key: [batch_size, seq_len, d_model]

    def forward(self, x):
        """
        :param x: (B, 3, H, W) e.g., (2, 3, 224, 224)
        :return: SA1:(B, 1, H/2, W/2), SA2:(B, 1, H/4, W/4), SA3:(B, 1, H/8, W/8)
        """
        # x: raw input (B, 3, H, W)
        # 第一次分解
        x_gray = x[:, [0], :, :]  # (B, 1, H, W)
        x_LH_1 = self.dwt(x_gray)  # (B, 4, H/2, W/2)
        x_L_1 = self.conv_l1(x_LH_1[:, [0], :, :])  # (B, 1, H/2, W/2) --> (B, 64)   Q
        x_H_1 = self.conv_h1(x_LH_1[:, 1:, :, :])  # (B, 3, H/2, W/2) --> (B, 64, H/2, W/2)   K
        bs, c, h, w = x_H_1.shape

        if self.norm_flag:
            x_L_1 = self.norm(x_L_1.flatten(2).permute(0, 2, 1)).permute(0, 2,
                                                                         1)  # (B,C,HW)-->(B, HW, C)-->layernorm-->(B,C,HW)
            x_H_1 = self.norm(x_H_1.flatten(2).permute(0, 2, 1))  # (B,C,HW)-->(B, HW, C)
        else:
            x_L_1 = x_L_1.unsqueeze(2)  # (B,C,1)
            x_H_1 = x_H_1.flatten(2).permute(0, 2, 1)  # (B,C,HW)-->(B, HW, C)

        correlation1 = torch.bmm(x_H_1, x_L_1)  # (B,HW,1)
        correlation1 = torch.reshape(correlation1, (bs, 1, h, w))  # (B,HW,1) --> (B,1,H,W)
        correlation1 = self.sigmoid(self.conv_sa1(correlation1))
        SA1 = correlation1  # (B, 1, H/2, W/2)

        # 二次分解
        x_LH_2 = self.dwt(x_LH_1[:, [0], :, :])  # (B, 4, H/4, W/4)
        x_L_2 = self.conv_l2(x_LH_2[:, [0], :, :])  # (B, 1, H/4, W/4) --> (B, 64)   Q
        x_H_2 = self.conv_h2(x_LH_2[:, 1:, :, :])  # (B, 3, H/4, W/4) --> (B, 64, H/4, W/4)
        bs, c, h, w = x_H_2.shape
        if self.norm_flag:
            x_L_2 = self.norm(x_L_2.flatten(2).permute(0, 2, 1)).permute(0, 2, 1)
            x_H_2 = self.norm(x_H_2.flatten(2).permute(0, 2, 1))
        else:
            x_L_2 = x_L_2.unsqueeze(2)
            x_H_2 = x_H_2.flatten(2).permute(0, 2, 1)

        correlation2 = torch.bmm(x_H_2, x_L_2)
        correlation2 = torch.reshape(correlation2, (bs, 1, h, w))
        correlation2 = self.sigmoid(self.conv_sa2(correlation2))
        SA2 = correlation2  # (B, 1, H/4, W/4)

        # 三次分解
        x_LH_3 = self.dwt(x_LH_2[:, [0], :, :])  # (B, 4, H/8, W/8)
        x_L_3 = self.conv_l3(x_LH_3[:, [0], :, :])  # (B, 1, H/8, W/8) --> (B, 64)   Q
        x_H_3 = self.conv_h3(x_LH_3[:, 1:, :, :])  # (B, 3, H/8, W/8) --> (B, 64, H/8, W/8)   K
        bs, c, h, w = x_H_3.shape
        if self.norm_flag:
            x_L_3 = self.norm(x_L_3.flatten(2).permute(0, 2, 1)).permute(0, 2, 1)
            x_H_3 = self.norm(x_H_3.flatten(2).permute(0, 2, 1))
        else:
            x_L_3 = x_L_3.unsqueeze(2)
            x_H_3 = x_H_3.flatten(2).permute(0, 2, 1)

        correlation3 = torch.bmm(x_H_3, x_L_3)
        correlation3 = torch.reshape(correlation3, (bs, 1, h, w))
        correlation3 = self.sigmoid(self.conv_sa3(correlation3))
        SA3 = correlation3  # (B, 1, H/8, W/8)

        return SA1, SA2, SA3


# ---------------------------------------------------------------------------
# --- 步骤 2: 从 CMUNet.py 复制的代码 ---
# ---------------------------------------------------------------------------

class MSAG(nn.Module):
    """
    Multi-scale attention gate
    来源: fenghetan9/.../src/network/conv_based/CMUNet.py
    """

    def __init__(self, channel):
        super(MSAG, self).__init__()
        self.channel = channel
        self.pointwiseConv = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(self.channel),
        )
        self.ordinaryConv = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm2d(self.channel),
        )
        self.dilationConv = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel_size=3, padding=2, stride=1, dilation=2, bias=True),
            nn.BatchNorm2d(self.channel),
        )
        self.voteConv = nn.Sequential(
            nn.Conv2d(self.channel * 3, self.channel, kernel_size=(1, 1)),
            nn.BatchNorm2d(self.channel),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.pointwiseConv(x)
        x2 = self.ordinaryConv(x)
        x3 = self.dilationConv(x)
        _x = self.relu(torch.cat((x1, x2, x3), dim=1))
        _x = self.voteConv(_x)
        x = x + x * _x
        return x


class Residual(nn.Module):
    """
    来源: fenghetan9/.../src/network/conv_based/CMUNet.py
    """

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class ConvMixerBlock(nn.Module):
    """
    来源: fenghetan9/.../src/network/conv_based/CMUNet.py
    """

    def __init__(self, dim=1024, depth=7, k=7):
        super(ConvMixerBlock, self).__init__()
        self.block = nn.Sequential(
            *[nn.Sequential(
                Residual(nn.Sequential(
                    # deep wise
                    nn.Conv2d(dim, dim, kernel_size=(k, k), groups=dim, padding=(k // 2, k // 2)),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(dim)
            ) for i in range(depth)]
        )

    def forward(self, x):
        x = self.block(x)
        return x


class conv_block(nn.Module):
    """
    来源: fenghetan9/.../src/network/conv_based/CMUNet.py
    """

    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    来源: fenghetan9/.../src/network/conv_based/CMUNet.py
    """

    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


# ---------------------------------------------------------------------------
# --- 步骤 3: 结合 MSHFFA 的新 CMUNet 模型 ---
# ---------------------------------------------------------------------------

class CMUNet_MSHFFA(nn.Module):
    """
    这是 CMUNet 与 MSHFFA 模块结合的新模型。
    MSHFFA 模块 (DWT_spatial_attention3) 被用于在编码器路径中生成
    多尺度的增强图 (AMs)，并应用于相应尺度的特征图 (x2, x3, x4)
    以增强高频细节并抑制噪声。
    """

    def __init__(self, img_ch=3, output_ch=1, l=7, k=7):
        """
        Args:
            img_ch : input channel.
            output_ch: output channel.
            l: number of convMixer layers
            k: kernal size of convMixer
        """
        super(CMUNet_MSHFFA, self).__init__()

        # --- MSHFFA 模块初始化 ---
        # 假设隐藏维度为 64, 与 MSHFFA 项目中的 ResNet 示例一致
        self.dwt_sa = DWT_spatial_attention3(hidden_d=64)
        # --- MSHFFA 结束 ---

        # Encoder (与原 CMUNet 相同)
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)
        self.ConvMixer = ConvMixerBlock(dim=1024, depth=l, k=k)

        # Decoder (与原 CMUNet 相同)
        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = conv_block(ch_in=512 * 2, ch_out=512)
        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=256 * 2, ch_out=256)
        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=128 * 2, ch_out=128)
        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=64 * 2, ch_out=64)
        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

        # Skip-connection (与原 CMUNet 相同)
        self.msag4 = MSAG(512)
        self.msag3 = MSAG(256)
        self.msag2 = MSAG(128)
        self.msag1 = MSAG(64)

    def forward(self, x):
        # 假定输入 x 的尺寸为 (B, 3, 256, 256)

        # --- MSHFFA 集成: 计算多尺度增强图 (AMs) ---
        # sa1: (B, 1, 128, 128)
        # sa2: (B, 1, 64, 64)
        # sa3: (B, 1, 32, 32)
        sa1, sa2, sa3 = self.dwt_sa(x)
        # --- MSHFFA 结束 ---

        # --- 编码器路径 ---
        x1 = self.Conv1(x)  # (B, 64, 256, 256)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)  # (B, 128, 128, 128)
        x2 = x2 * sa1  # <-- 应用 MSHFFA 尺度 1

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)  # (B, 256, 64, 64)
        x3 = x3 * sa2  # <-- 应用 MSHFFA 尺度 2

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)  # (B, 512, 32, 32)
        x4 = x4 * sa3  # <-- 应用 MSHFFA 尺度 3

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)  # (B, 1024, 16, 16)
        x5 = self.ConvMixer(x5)

        # Skip-connection (使用增强后的特征图)
        x4 = self.msag4(x4)
        x3 = self.msag3(x3)
        x2 = self.msag2(x2)
        x1 = self.msag1(x1)

        # --- 解码器路径 ---
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1