"""
HydroKAN-Net: Spectral-Gated Kolmogorov-Arnold Networks with Boundary-Aware
Cross-Scale Aggregation for Flood Segmentation in Unmanned Aerial Vehicle Imagery

Baseline: UKAN (Tokenized KAN U-Net)
Three novelties:
  (1) SG-KAB  - Spectral-Gated KAN Block (replaces KANBlock)
  (2) BCSA    - Boundary-aware Cross-Scale Aggregation bridge (replaces additive skip)
  (3) WAFL    - Water-Aware Frequency Loss with deep supervision

Author: (your name)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import init


def to_2tuple(x):
    if isinstance(x, (int, float)):
        return (int(x), int(x))
    return x


# ---------------------------------------------------------------------------
# DropPath
# ---------------------------------------------------------------------------
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x / keep_prob * random_tensor


# ---------------------------------------------------------------------------
# KANLinear (B-spline based linear layer) - retained from baseline
# ---------------------------------------------------------------------------
class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3,
                 scale_noise=0.1, scale_base=1.0, scale_spline=1.0,
                 enable_standalone_scale_spline=True, base_activation=nn.SiLU,
                 grid_eps=0.02, grid_range=[-1, 1]):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = ((torch.arange(-spline_order, grid_size + spline_order + 1) * h
                 + grid_range[0]).expand(in_features, -1).contiguous())
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order))
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.Tensor(out_features, in_features))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = ((torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                      - 1 / 2) * self.scale_noise / self.grid_size)
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(self.grid.T[self.spline_order:-self.spline_order], noise))
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x):
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (((x - grid[:, :-(k + 1)]) / (grid[:, k:-1] - grid[:, :-(k + 1)])
                      * bases[:, :, :-1])
                     + ((grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)])
                        * bases[:, :, 1:]))
        return bases.contiguous()

    def curve2coeff(self, x, y):
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        return solution.permute(2, 0, 1).contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline else 1.0)

    def forward(self, x):
        assert x.dim() == 2 and x.size(1) == self.in_features
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1))
        return base_output + spline_output

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        l1_fake = self.spline_weight.abs().mean(-1)
        reg_act = l1_fake.sum()
        p = l1_fake / reg_act
        reg_ent = -torch.sum(p * p.log())
        return regularize_activation * reg_act + regularize_entropy * reg_ent


# ---------------------------------------------------------------------------
# Depthwise conv + BN + ReLU (token spatial mixing) - retained
# ---------------------------------------------------------------------------
class DW_bn_relu(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.relu(self.bn(self.dwconv(x)))
        return x.flatten(2).transpose(1, 2)


# ===========================================================================
# NOVELTY 1 :  Spectral-Gated KAN Block  (SG-KAB)
# ===========================================================================
class SpectralGate(nn.Module):
    """
    Frequency-domain gating tuned to water's spectral signature.

    Tokens [B, N, C] -> reshape to [B, C, H, W] -> 2D rFFT.
    A learnable per-channel gate (low_gate, high_gate) re-weights the
    low-frequency (water-homogeneous) and high-frequency (edge/clutter)
    spectral bands separately, then inverse-FFTs back. A radial mask
    separates the two bands. The high-frequency band is preserved through
    a residual path so boundary cues are not destroyed.
    """
    def __init__(self, dim, cutoff_ratio=0.25):
        super().__init__()
        self.dim = dim
        self.cutoff_ratio = cutoff_ratio
        # learnable complex-valued gates for low / high bands (per channel)
        self.low_gate = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.high_gate = nn.Parameter(torch.ones(1, dim, 1, 1) * 0.5)
        self.proj = nn.Conv2d(dim, dim, 1, bias=True)

    def _radial_mask(self, H, Wf, device):
        # low-frequency mask in the rFFT grid (DC at corner-> we build centered)
        yy = torch.linspace(-1, 1, H, device=device).view(H, 1)
        xx = torch.linspace(0, 1, Wf, device=device).view(1, Wf)  # rfft -> half width
        radius = torch.sqrt(yy ** 2 + xx ** 2)
        radius = radius / radius.max()
        low_mask = (radius <= self.cutoff_ratio).float()
        return low_mask.unsqueeze(0).unsqueeze(0)  # [1,1,H,Wf]

    def forward(self, x, H, W):
        B, N, C = x.shape
        x_img = x.transpose(1, 2).reshape(B, C, H, W)

        # 2D real FFT
        fft = torch.fft.rfft2(x_img, norm='ortho')          # [B,C,H,Wf] complex
        Wf = fft.shape[-1]
        low_mask = self._radial_mask(H, Wf, x.device)        # [1,1,H,Wf]
        high_mask = 1.0 - low_mask

        low = fft * low_mask
        high = fft * high_mask
        # apply learnable per-channel band gating
        fft_gated = low * self.low_gate + high * self.high_gate

        x_filt = torch.fft.irfft2(fft_gated, s=(H, W), norm='ortho')  # [B,C,H,W]
        x_filt = self.proj(x_filt)
        out = x_filt.flatten(2).transpose(1, 2)              # [B,N,C]
        return out


class SGKAB_Layer(nn.Module):
    """Spectral-Gated KAN layer: SpectralGate -> KAN spline mapping -> DWconv."""
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 drop=0., no_kan=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features

        self.spectral_gate = SpectralGate(in_features)

        kan_cfg = dict(grid_size=5, spline_order=3, scale_noise=0.1, scale_base=1.0,
                       scale_spline=1.0, base_activation=nn.SiLU,
                       grid_eps=0.02, grid_range=[-1, 1])
        if not no_kan:
            self.fc1 = KANLinear(in_features, hidden_features, **kan_cfg)
            self.fc2 = KANLinear(hidden_features, out_features, **kan_cfg)
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)

        self.dwconv_1 = DW_bn_relu(hidden_features)
        self.dwconv_2 = DW_bn_relu(out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        # spectral gating (residual: keep original tokens too)
        x = x + self.spectral_gate(x, H, W)

        x = self.fc1(x.reshape(B * N, C))
        x = x.reshape(B, N, -1).contiguous()
        x = self.dwconv_1(x, H, W)

        x = self.fc2(x.reshape(B * N, x.shape[-1]))
        x = x.reshape(B, N, -1).contiguous()
        x = self.dwconv_2(x, H, W)
        return x


class SGKAB(nn.Module):
    """Spectral-Gated KAN Block: norm -> SGKAB_Layer -> residual + droppath."""
    def __init__(self, dim, drop=0., drop_path=0., norm_layer=nn.LayerNorm, no_kan=False):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm = norm_layer(dim)
        self.layer = SGKAB_Layer(in_features=dim, hidden_features=dim,
                                 drop=drop, no_kan=no_kan)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        return x + self.drop_path(self.layer(self.norm(x), H, W))


# ===========================================================================
# NOVELTY 2 :  Boundary-aware Cross-Scale Aggregation bridge (BCSA)
# ===========================================================================
class BCSA(nn.Module):
    """
    Replaces the additive skip connection.

    Inputs:
        enc  : encoder skip feature           [B, C, H, W]
        dec  : upsampled decoder feature       [B, C, H, W]
        ctx  : global cross-scale context vec  [B, Cctx]  (optional)
    Steps:
        1. Water-edge attention from a Sobel gradient of `enc`.
        2. Channel recalibration of `enc` using a global descriptor that
           fuses enc + dec + cross-scale context (channel gate).
        3. Spatial gate (edge map) applied to the recalibrated skip.
        4. Fuse with decoder feature.
    """
    def __init__(self, channels, ctx_dim=0):
        super().__init__()
        self.channels = channels

        # fixed Sobel kernels (registered, not learned) for water-edge cue
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = sobel_x.t().contiguous()
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

        # spatial gate conv (from edge magnitude -> attention map)
        self.edge_conv = nn.Sequential(
            nn.Conv2d(1, 1, 3, padding=1, bias=True),
            nn.Sigmoid())

        # channel gate MLP : descriptor = GAP(enc) + GAP(dec) (+ ctx)
        gate_in = channels * 2 + ctx_dim
        self.channel_gate = nn.Sequential(
            nn.Linear(gate_in, max(channels // 4, 8)),
            nn.GELU(),
            nn.Linear(max(channels // 4, 8), channels),
            nn.Sigmoid())

        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True))

    def water_edge_map(self, enc):
        # grayscale-style channel mean, then Sobel magnitude
        g = enc.mean(dim=1, keepdim=True)
        gx = F.conv2d(g, self.sobel_x, padding=1)
        gy = F.conv2d(g, self.sobel_y, padding=1)
        mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
        return self.edge_conv(mag)            # [B,1,H,W] in [0,1]

    def forward(self, enc, dec, ctx=None):
        B, C, H, W = enc.shape

        # ---- channel gate ----
        desc = [F.adaptive_avg_pool2d(enc, 1).flatten(1),
                F.adaptive_avg_pool2d(dec, 1).flatten(1)]
        if ctx is not None:
            desc.append(ctx)
        desc = torch.cat(desc, dim=1)
        ch_gate = self.channel_gate(desc).view(B, C, 1, 1)
        enc_ch = enc * ch_gate

        # ---- spatial (water-edge) gate ----
        edge = self.water_edge_map(enc)       # [B,1,H,W]
        enc_sp = enc_ch * (1.0 + edge)        # boost interface, keep base

        # ---- fuse with decoder ----
        out = self.fuse(torch.cat([enc_sp, dec], dim=1))
        return out


class CrossScaleContext(nn.Module):
    """
    Builds a single global cross-scale context vector from a list of encoder
    feature maps (different channel widths). Each is GAP'd, projected to a
    shared latent dim, concatenated, and passed through an MLP. The output is
    shared by all BCSA bridges (cross-scale, not same-scale only).
    """
    def __init__(self, in_channels_list, latent=128, out_dim=128):
        super().__init__()
        self.projs = nn.ModuleList([nn.Linear(c, latent) for c in in_channels_list])
        self.mlp = nn.Sequential(
            nn.Linear(latent * len(in_channels_list), out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim))

    def forward(self, feats):
        vecs = []
        for proj, f in zip(self.projs, feats):
            v = F.adaptive_avg_pool2d(f, 1).flatten(1)   # [B, c]
            vecs.append(proj(v))                          # [B, latent]
        ctx = torch.cat(vecs, dim=1)
        return self.mlp(ctx)                              # [B, out_dim]


# ---------------------------------------------------------------------------
# Patch embedding / conv blocks - retained from baseline
# ---------------------------------------------------------------------------
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class D_ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1), nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


# ===========================================================================
# HydroKAN-Net
# ===========================================================================
class HydroKANNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, img_size=256,
                 embed_dims=[256, 320, 512], no_kan=False,
                 drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision
        kan_dim = embed_dims[0]

        # ---- Encoder (conv stem) ----
        self.encoder1 = ConvLayer(input_channels, kan_dim // 8)
        self.encoder2 = ConvLayer(kan_dim // 8, kan_dim // 4)
        self.encoder3 = ConvLayer(kan_dim // 4, kan_dim)

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])
        self.dnorm3 = norm_layer(embed_dims[1])
        self.dnorm4 = norm_layer(embed_dims[0])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # ---- Tokenized SG-KAB stages ----
        self.block1 = nn.ModuleList([SGKAB(embed_dims[1], drop=drop_rate,
                       drop_path=dpr[i], norm_layer=norm_layer, no_kan=no_kan)
                       for i in range(depths[0])])
        self.block2 = nn.ModuleList([SGKAB(embed_dims[2], drop=drop_rate,
                       drop_path=dpr[sum(depths[:1]) + i], norm_layer=norm_layer,
                       no_kan=no_kan) for i in range(depths[1])])
        self.dblock1 = nn.ModuleList([SGKAB(embed_dims[1], drop=drop_rate,
                        drop_path=dpr[sum(depths[:2]) + i], norm_layer=norm_layer,
                        no_kan=no_kan) for i in range(depths[2])])
        self.dblock2 = nn.ModuleList([SGKAB(embed_dims[0], drop=drop_rate,
                        drop_path=dpr[i], norm_layer=norm_layer, no_kan=no_kan)
                        for i in range(depths[0])])

        self.patch_embed3 = PatchEmbed(img_size // 4, 3, 2, embed_dims[0], embed_dims[1])
        self.patch_embed4 = PatchEmbed(img_size // 8, 3, 2, embed_dims[1], embed_dims[2])

        # ---- Decoder convs ----
        self.decoder1 = D_ConvLayer(embed_dims[2], embed_dims[1])
        self.decoder2 = D_ConvLayer(embed_dims[1], embed_dims[0])
        self.decoder3 = D_ConvLayer(embed_dims[0], embed_dims[0] // 4)
        self.decoder4 = D_ConvLayer(embed_dims[0] // 4, embed_dims[0] // 8)
        self.decoder5 = D_ConvLayer(embed_dims[0] // 8, embed_dims[0] // 8)

        # ---- Cross-scale context (from t2,t3,t4) ----
        self.cross_ctx = CrossScaleContext(
            in_channels_list=[kan_dim // 4, kan_dim, embed_dims[1]],
            latent=128, out_dim=128)

        # ---- BCSA bridges (one per fused skip) ----
        self.bcsa_t4 = BCSA(embed_dims[1], ctx_dim=128)
        self.bcsa_t3 = BCSA(embed_dims[0], ctx_dim=128)
        self.bcsa_t2 = BCSA(embed_dims[0] // 4, ctx_dim=128)
        self.bcsa_t1 = BCSA(embed_dims[0] // 8, ctx_dim=128)

        # ---- Heads ----
        self.final = nn.Conv2d(embed_dims[0] // 8, num_classes, 1)
        if self.deep_supervision:
            self.ds1 = nn.Conv2d(embed_dims[1], num_classes, 1)
            self.ds2 = nn.Conv2d(embed_dims[0], num_classes, 1)
            self.ds3 = nn.Conv2d(embed_dims[0] // 4, num_classes, 1)

    def forward(self, x):
        B = x.shape[0]
        H0, W0 = x.shape[2], x.shape[3]

        # ---- Encoder ----
        out = F.relu(F.max_pool2d(self.encoder1(x), 2, 2)); t1 = out
        out = F.relu(F.max_pool2d(self.encoder2(out), 2, 2)); t2 = out
        out = F.relu(F.max_pool2d(self.encoder3(out), 2, 2)); t3 = out

        # ---- SG-KAB stage 1 ----
        out, H, W = self.patch_embed3(out)
        for blk in self.block1:
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        # ---- Bottleneck SG-KAB stage 2 ----
        out, H, W = self.patch_embed4(out)
        for blk in self.block2:
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # ---- Cross-scale global context (shared) ----
        ctx = self.cross_ctx([t2, t3, t4])

        ds_outputs = []

        # ---- Decoder 1 + BCSA(t4) ----
        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=2, mode='bilinear',
                                   align_corners=False))
        out = self.bcsa_t4(t4, out, ctx)
        if self.deep_supervision:
            ds_outputs.append(F.interpolate(self.ds1(out), size=(H0, W0),
                              mode='bilinear', align_corners=False))
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.dblock1:
            out = blk(out, H, W)
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # ---- Decoder 2 + BCSA(t3) ----
        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=2, mode='bilinear',
                                   align_corners=False))
        out = self.bcsa_t3(t3, out, ctx)
        if self.deep_supervision:
            ds_outputs.append(F.interpolate(self.ds2(out), size=(H0, W0),
                              mode='bilinear', align_corners=False))
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.dblock2:
            out = blk(out, H, W)
        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # ---- Decoder 3 + BCSA(t2) ----
        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=2, mode='bilinear',
                                   align_corners=False))
        out = self.bcsa_t2(t2, out, ctx)
        if self.deep_supervision:
            ds_outputs.append(F.interpolate(self.ds3(out), size=(H0, W0),
                              mode='bilinear', align_corners=False))

        # ---- Decoder 4 + BCSA(t1) ----
        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=2, mode='bilinear',
                                   align_corners=False))
        out = self.bcsa_t1(t1, out, ctx)

        # ---- Decoder 5 -> final ----
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=2, mode='bilinear',
                                   align_corners=False))
        main = self.final(out)
        if main.shape[2:] != (H0, W0):
            main = F.interpolate(main, size=(H0, W0), mode='bilinear', align_corners=False)

        if self.deep_supervision and self.training:
            return main, ds_outputs
        return main

    def regularization_loss(self, ra=1.0, re=1.0):
        total = 0.
        for m in self.modules():
            if isinstance(m, KANLinear):
                total += m.regularization_loss(ra, re)
        return total


# ===========================================================================
# NOVELTY 3 :  Water-Aware Frequency Loss (WAFL)
# ===========================================================================
class WaterAwareFrequencyLoss(nn.Module):
    """
    L = w_dice*Dice + w_bce*BCE + w_freq*FreqConsistency
    FreqConsistency = L1 between FFT-magnitude spectra of sigmoid(pred) and gt.
    Encourages spectrally smooth (non-fragmented) water masks.
    Deep supervision: same loss applied to each auxiliary head with decaying weights.
    """
    def __init__(self, w_dice=0.5, w_bce=0.3, w_freq=0.2,
                 ds_weights=(0.5, 0.3, 0.15)):
        super().__init__()
        self.w_dice, self.w_bce, self.w_freq = w_dice, w_bce, w_freq
        self.ds_weights = ds_weights
        self.bce = nn.BCEWithLogitsLoss()

    def dice_loss(self, logits, target, eps=1e-6):
        prob = torch.sigmoid(logits)
        num = 2 * (prob * target).sum(dim=(2, 3)) + eps
        den = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps
        return (1 - num / den).mean()

    def freq_loss(self, logits, target):
        prob = torch.sigmoid(logits)
        pf = torch.fft.rfft2(prob, norm='ortho').abs()
        tf = torch.fft.rfft2(target, norm='ortho').abs()
        return F.l1_loss(pf, tf)

    def single(self, logits, target):
        return (self.w_dice * self.dice_loss(logits, target)
                + self.w_bce * self.bce(logits, target)
                + self.w_freq * self.freq_loss(logits, target))

    def forward(self, outputs, target):
        if isinstance(outputs, tuple):
            main, ds_list = outputs
        else:
            main, ds_list = outputs, []
        loss = self.single(main, target)
        for w, ds in zip(self.ds_weights, ds_list):
            loss = loss + w * self.single(ds, target)
        return loss


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = HydroKANNet(num_classes=1, input_channels=3, img_size=256,
                        embed_dims=[256, 320, 512], depths=[1, 1, 1],
                        deep_supervision=True)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"HydroKAN-Net trainable params: {n_params/1e6:.2f} M")

    x = torch.randn(2, 3, 256, 256)
    y = (torch.rand(2, 1, 256, 256) > 0.5).float()

    model.train()
    out = model(x)
    crit = WaterAwareFrequencyLoss()
    loss = crit(out, y) + 1e-5 * model.regularization_loss()
    loss.backward()
    print("train forward+backward OK | loss =", loss.item())
    if isinstance(out, tuple):
        print("main:", out[0].shape, "| #ds heads:", len(out[1]))

    model.eval()
    with torch.no_grad():
        pred = model(x)
    print("eval output:", pred.shape)
