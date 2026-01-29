import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------
# 基础组件 (从 CMUNeXt 复用)
# --------------------------

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)


class fusion_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(fusion_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, groups=2, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(ch_in),
            nn.Conv2d(ch_in, ch_out * 4, kernel_size=(1, 1)),
            nn.GELU(),
            nn.BatchNorm2d(ch_out * 4),
            nn.Conv2d(ch_out * 4, ch_out, kernel_size=(1, 1)),
            nn.GELU(),
            nn.BatchNorm2d(ch_out)
        )

    def forward(self, x):
        return self.conv(x)


class CMUNeXtBlock(nn.Module):
    def __init__(self, ch_in, ch_out, depth=1, k=3):
        super(CMUNeXtBlock, self).__init__()
        self.block = nn.Sequential(
            *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(ch_in, ch_in, kernel_size=(k, k), groups=ch_in, padding=(k // 2, k // 2)),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in)
                )),
                nn.Conv2d(ch_in, ch_in * 4, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_in * 4),
                nn.Conv2d(ch_in * 4, ch_in, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_in)
            ) for _ in range(depth)]
        )
        self.up = conv_block(ch_in, ch_out)

    def forward(self, x):
        x = self.block(x)
        x = self.up(x)
        return x


# --------------------------
# 核心创新模块: 频域去噪 (FFT)
# --------------------------

class FFTLowPassDenoise2D(nn.Module):
    """
    Differentiable FFT low-pass filtering for feature maps.

    注意：
    - 为 AMP/fp16/bf16 兼容，mask dtype 会对齐到 FFT 输出的 real dtype（通常是 float32）
    - cutoff_ratio 是相对于频域最大径向距离 r_max 的比例（r_max 采用对角线半径）
    """
    def __init__(self, cutoff_ratio: float = 0.35):
        super().__init__()
        if not (0.0 < cutoff_ratio <= 0.5):
            raise ValueError("cutoff_ratio must be in (0, 0.5].")
        self.cutoff_ratio = cutoff_ratio

    @staticmethod
    def _lowpass_mask(h: int, w: int, cutoff_ratio: float, device, dtype):
        # centered coordinates: [-h/2, h/2), [-w/2, w/2)
        yy = torch.arange(h, device=device, dtype=dtype) - (h / 2.0)
        xx = torch.arange(w, device=device, dtype=dtype) - (w / 2.0)
        yy = yy.view(h, 1)
        xx = xx.view(1, w)

        rr = torch.sqrt(yy * yy + xx * xx)  # [H, W]

        # r_max: diagonal radius
        r_max_sq = (h / 2.0) ** 2 + (w / 2.0) ** 2
        r_max = torch.sqrt(torch.as_tensor(r_max_sq, device=device, dtype=dtype))

        r_cut = cutoff_ratio * r_max
        mask = (rr <= r_cut).to(dtype=dtype)
        return mask  # [H, W]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        # FFT (complex)
        X = torch.fft.fft2(x, dim=(-2, -1))
        X = torch.fft.fftshift(X, dim=(-2, -1))

        # AMP/dtype-safe: align mask dtype to FFT output dtype
        mask_dtype = X.real.dtype
        mask = self._lowpass_mask(h, w, self.cutoff_ratio, device=x.device, dtype=mask_dtype)
        mask = mask.view(1, 1, h, w)

        # Apply low-pass mask (broadcast over B,C)
        X_filtered = X * mask

        # iFFT back
        X_filtered = torch.fft.ifftshift(X_filtered, dim=(-2, -1))
        x_rec = torch.fft.ifft2(X_filtered, dim=(-2, -1)).real

        # Ensure output dtype matches input dtype (optional but often helpful)
        # Note: if FFT internally upcasts, this will cast back for consistency.
        if x_rec.dtype != x.dtype:
            x_rec = x_rec.to(dtype=x.dtype)

        return x_rec


# --------------------------
# 主网络: CMUNeXt_FFT
# --------------------------

class CMUNeXt_FFT(nn.Module):
    def __init__(
        self,
        input_channel=3,
        num_classes=1,
        dims=[16, 32, 128, 160, 256],
        depths=[1, 1, 1, 3, 1],
        kernels=[3, 3, 7, 7, 7],
        fft_cutoff_ratio=0.35,
        # 默认最稳：只对 Stage3 做 FFT 去噪（256输入下更保边缘、更不容易过平滑）
        fft_apply_stages=(3,),

        #第二个消融：中等去噪
    #   fft_apply_stages=(3, 4)

        #第三个消融：强去噪
    #    fft_apply_stages=(2, 3, 4)
    ):
        super(CMUNeXt_FFT, self).__init__()

        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])

        # FFT Denoise
        self.fft_apply_stages = set(fft_apply_stages) if fft_apply_stages is not None else set()
        self.fft_denoise = FFTLowPassDenoise2D(cutoff_ratio=fft_cutoff_ratio)

        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])

        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])

        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])

        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])

        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def _maybe_fft(self, x: torch.Tensor, stage_idx: int) -> torch.Tensor:
        if stage_idx in self.fft_apply_stages:
            return self.fft_denoise(x)
        return x

    def forward(self, x):
        # Stage 1
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x1 = self._maybe_fft(x1, 1)

        # Stage 2
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x2 = self._maybe_fft(x2, 2)

        # Stage 3 (默认在此做 FFT 去噪)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x3 = self._maybe_fft(x3, 3)

        # Stage 4
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x4 = self._maybe_fft(x4, 4)

        # Stage 5 (bottleneck) - usually no FFT here
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)

        # Decoder
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

        out = self.Conv_1x1(d2)
        return out
