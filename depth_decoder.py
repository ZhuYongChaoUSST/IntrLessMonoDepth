
from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from layers import *
from timm.models.layers import trunc_normal_

class CRF(nn.Module):
    def __init__(self, in_channels, out_channels, window_size):
        super(CRF, self).__init__()

        self.gr = 16

        self.window_size = window_size

        self.grad = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels, self.gr, kernel_size=3, padding=1))

        self.att = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels, self.gr, kernel_size=3, padding=1),
            nn.Sigmoid())

        num = window_size * window_size
        a = torch.ones(num, 1, num, dtype=torch.float16)  # 创建所有元素为1的矩阵

        eye = torch.eye(num, dtype=torch.float16)  # 生成单位矩阵
        for i in range(num):
            for j in range(1):
                a[i, j] -= eye[i]  # 将对角线元素置零

        self.register_buffer('a', a.unsqueeze(0))

        self.ins = nn.GroupNorm(1, self.gr)

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels // 2, self.gr, kernel_size=1, padding=0),
            nn.Sigmoid())

        self.post = nn.Sequential(
            nn.Conv2d(self.gr, out_channels, kernel_size=3, padding=1))

    def forward(self, x):
        n, c, H, W = x.shape

        x = window_partition(x, self.window_size)

        n, c, h, w = x.shape

        att = F.interpolate(x, scale_factor=(w, h), mode='bilinear', align_corners=True)

        att = self.att(att)

        se = self.se(x)

        x = self.grad(x)

        att = att.reshape(n * self.gr, 1, h * w, h * w).permute(0, 2, 1, 3)
        A = self.a * att

        A = A.reshape(n * self.gr, h * w, h * w)
        B = x.reshape(n * self.gr, h * w, 1)

        AB = torch.bmm(A, B)

        x = AB.reshape(n, self.gr, h, w)

        x = self.ins(x)

        x = se * x

        x = self.post(x)

        x = window_reverse(x, self.window_size, H, W)

        return x

def window_partition(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)

    """
    B, C, H, W = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, C, window_size, window_size)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, C, H, W)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, -1, H, W)
    return x


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True, window_size=4):
        super().__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'bilinear'
        self.scales = scales
        self.window_size = window_size

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = (self.num_ch_enc / 2).astype('int')

        self.device = torch.device("cuda")

        # decoder
        self.convs = OrderedDict()
        self.crf = OrderedDict()
        for i in range(2, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 2 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)
            self.crf[("crf", i, 0)] = CRF(self.num_ch_enc[i], num_ch_out, self.window_size).to(self.device)
            # upconv_1
            # num_ch_in = self.num_ch_dec[i]
            num_ch_in = 64
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)
        for s in self.scales:
            self.convs[("refine", s)] = nn.Conv2d(self.num_ch_dec[s], 64, kernel_size=3, padding=1)
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.decoder += nn.ModuleList(list(self.crf.values()))
        self.sigmoid = nn.Sigmoid()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, input_features):
        self.outputs = {}
        x = input_features[-1]
        for i in range(2, -1, -1):
            e = self.crf[("crf", i, 0)](input_features[i])
            x = self.convs[("upconv", i, 0)](x)
            x = x + e
            x = self.convs[("refine", i)](x)
            self.outputs[("feature", i)] = x
            x = [upsample(x)]

            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)

            if i in self.scales:
                f = upsample(self.convs[("dispconv", i)](x), mode='bilinear')
                self.outputs[("disp", i)] = self.sigmoid(f)
        return self.outputs