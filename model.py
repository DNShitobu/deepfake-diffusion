"""
DDPM Face Generation — Model Architecture
==========================================
Denoising Diffusion Probabilistic Model for photorealistic face synthesis.

Architecture:
  U-Net with:
    - Time-step sinusoidal embeddings
    - Residual blocks with GroupNorm + SiLU
    - Self-attention at 16x16 and 32x32 spatial resolutions
    - Skip connections from encoder to decoder

Forward process:
  q(x_t | x_{t-1}) = N(x_t; sqrt(1-beta_t) * x_{t-1}, beta_t * I)
  
Reverse process (learned):
  p_theta(x_{t-1} | x_t) = N(x_{t-1}; mu_theta(x_t, t), sigma_t^2 * I)
  where mu_theta is predicted by the U-Net

Loss:
  L_simple = E[||eps - eps_theta(sqrt(abar_t)*x0 + sqrt(1-abar_t)*eps, t)||^2]

Sampling: DDPM (1000 steps) or DDIM (50-250 steps, ~10x faster)

Reference:
  Ho et al. (2020) "Denoising Diffusion Probabilistic Models"
  Song et al. (2021) "Denoising Diffusion Implicit Models" (DDIM sampler)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Time Embedding ─────────────────────────────────────────────────────────────

def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Transformer-style sinusoidal positional encoding for timestep t."""
    assert dim % 2 == 0
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args  = t[:, None].float() * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(),
            nn.Linear(dim * 4, dim * 4),
        )
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(sinusoidal_embedding(t, self.dim))


# ── U-Net Building Blocks ─────────────────────────────────────────────────────

class ResNetBlock(nn.Module):
    """Residual block conditioned on time embedding."""
    def __init__(self, in_ch: int, out_ch: int, t_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.t_proj = nn.Sequential(nn.SiLU(), nn.Linear(t_dim, out_ch * 2))
        self.skip   = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        # Adaptive group norm conditioning: scale + shift from t
        t = self.t_proj(t_emb)[:, :, None, None]
        scale, shift = t.chunk(2, dim=1)
        h = F.silu(self.norm2(h) * (1 + scale) + shift)
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.skip(x)


class SelfAttention2D(nn.Module):
    """Multi-head self-attention for spatial feature maps."""
    def __init__(self, ch: int, n_heads: int = 8):
        super().__init__()
        self.norm  = nn.GroupNorm(8, ch)
        self.attn  = nn.MultiheadAttention(ch, n_heads, batch_first=True)
        self.proj  = nn.Conv2d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H*W).permute(0, 2, 1)
        h, _ = self.attn(h, h, h)
        h = h.permute(0, 2, 1).view(B, C, H, W)
        return x + self.proj(h)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim, use_attn=False):
        super().__init__()
        self.res1  = ResNetBlock(in_ch, out_ch, t_dim)
        self.res2  = ResNetBlock(out_ch, out_ch, t_dim)
        self.attn  = SelfAttention2D(out_ch) if use_attn else nn.Identity()
        self.down  = nn.Conv2d(out_ch, out_ch, 4, 2, 1)  # 2x downsample

    def forward(self, x, t):
        x = self.res1(x, t)
        x = self.res2(x, t)
        x = self.attn(x) if isinstance(self.attn, SelfAttention2D) else self.attn
        return self.down(x), x  # (downsampled, skip)


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, t_dim, use_attn=False):
        super().__init__()
        self.up    = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
        )
        self.res1  = ResNetBlock(out_ch + skip_ch, out_ch, t_dim)
        self.res2  = ResNetBlock(out_ch, out_ch, t_dim)
        self.attn  = SelfAttention2D(out_ch) if use_attn else nn.Identity()

    def forward(self, x, skip, t):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, t)
        x = self.res2(x, t)
        return self.attn(x) if isinstance(self.attn, SelfAttention2D) else x


# ── U-Net ─────────────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    U-Net that predicts the noise epsilon added at timestep t.
    Input:  noisy image x_t (B, 3, H, W) + timestep t (B,)
    Output: predicted noise eps_theta (B, 3, H, W)
    
    Channel schedule: [64, 128, 256, 512]
    Attention at 16x16 and 32x32 resolutions.
    """
    def __init__(self, img_ch=3, base_ch=64, ch_mult=(1,2,4,8), t_dim=256):
        super().__init__()
        chs = [base_ch * m for m in ch_mult]  # [64, 128, 256, 512]
        self.t_emb = TimeEmbedding(t_dim)
        self.stem  = nn.Conv2d(img_ch, chs[0], 3, 1, 1)

        # Encoder
        # 256->128 (no attn), 128->64 (no attn), 64->32 (attn), 32->16 (attn)
        self.downs = nn.ModuleList([
            DownBlock(chs[0], chs[0], t_dim*4, use_attn=False),
            DownBlock(chs[0], chs[1], t_dim*4, use_attn=False),
            DownBlock(chs[1], chs[2], t_dim*4, use_attn=True),
            DownBlock(chs[2], chs[3], t_dim*4, use_attn=True),
        ])
        # Bottleneck (16x16)
        self.mid_res1 = ResNetBlock(chs[3], chs[3], t_dim*4)
        self.mid_attn = SelfAttention2D(chs[3])
        self.mid_res2 = ResNetBlock(chs[3], chs[3], t_dim*4)

        # Decoder
        self.ups = nn.ModuleList([
            UpBlock(chs[3], chs[3], chs[3], t_dim*4, use_attn=True),
            UpBlock(chs[3], chs[2], chs[2], t_dim*4, use_attn=True),
            UpBlock(chs[2], chs[1], chs[1], t_dim*4, use_attn=False),
            UpBlock(chs[1], chs[0], chs[0], t_dim*4, use_attn=False),
        ])
        self.out = nn.Sequential(
            nn.GroupNorm(8, chs[0]), nn.SiLU(),
            nn.Conv2d(chs[0], img_ch, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.t_emb(t)
        x = self.stem(x)
        skips = []
        for down in self.downs:
            x, skip = down(x, t_emb)
            skips.append(skip)
        x = self.mid_res1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_res2(x, t_emb)
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip, t_emb)
        return self.out(x)


# ── Diffusion Process ─────────────────────────────────────────────────────────

class GaussianDiffusion(nn.Module):
    """
    Wraps the U-Net with the diffusion forward/reverse process.
    Implements both DDPM (slow) and DDIM (fast) sampling.
    """
    def __init__(self, model: UNet, T: int = 1000,
                 beta_start: float = 1e-4, beta_end: float = 0.02):
        super().__init__()
        self.model = model
        self.T     = T
        # Linear beta schedule
        betas  = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas",        betas)
        self.register_buffer("alphas",       alphas)
        self.register_buffer("alpha_bar",    alpha_bar)
        self.register_buffer("sqrt_abar",    alpha_bar.sqrt())
        self.register_buffer("sqrt_1m_abar", (1.0 - alpha_bar).sqrt())

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor = None):
        """Forward diffusion: q(x_t | x_0) — add noise at timestep t."""
        if noise is None:
            noise = torch.randn_like(x0)
        s_ab  = self.sqrt_abar[t][:, None, None, None]
        s_1ab = self.sqrt_1m_abar[t][:, None, None, None]
        return s_ab * x0 + s_1ab * noise, noise

    def p_loss(self, x0: torch.Tensor) -> torch.Tensor:
        """Training loss: predict the noise added at random timestep t."""
        B = x0.size(0)
        t = torch.randint(0, self.T, (B,), device=x0.device)
        x_noisy, noise = self.q_sample(x0, t)
        pred_noise = self.model(x_noisy, t)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def ddpm_sample(self, shape, device) -> torch.Tensor:
        """Standard DDPM reverse sampling (slow: T=1000 steps)."""
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.T)):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            pred_noise = self.model(x, t_batch)
            alpha_t    = self.alphas[t]
            alpha_bar_t = self.alpha_bar[t]
            beta_t     = self.betas[t]
            # Predicted x0
            x0_hat = (x - (1-alpha_bar_t).sqrt() * pred_noise) / alpha_bar_t.sqrt()
            x0_hat = x0_hat.clamp(-1, 1)
            # Posterior mean
            mean = (alpha_t.sqrt() * (1 - alpha_bar_t/alpha_t) * x +
                    (alpha_bar_t/alpha_t).sqrt() * (1-alpha_t) * x0_hat) / (1-alpha_bar_t)
            if t > 0:
                var  = beta_t * (1 - alpha_bar_t/alpha_t) / (1 - alpha_bar_t)
                x = mean + var.sqrt() * torch.randn_like(x)
            else:
                x = mean
        return x.clamp(-1, 1)

    @torch.no_grad()
    def ddim_sample(self, shape, device, n_steps: int = 50) -> torch.Tensor:
        """DDIM sampling — ~10x faster with comparable quality."""
        seq = torch.linspace(0, self.T-1, n_steps, dtype=torch.long)
        x   = torch.randn(shape, device=device)
        for i in reversed(range(1, len(seq))):
            t_cur  = seq[i];   t_prev = seq[i-1]
            t_b    = torch.full((shape[0],), t_cur, device=device, dtype=torch.long)
            pred_n = self.model(x, t_b)
            ab_t   = self.alpha_bar[t_cur]
            ab_p   = self.alpha_bar[t_prev]
            x0_hat = (x - (1-ab_t).sqrt() * pred_n) / ab_t.sqrt()
            x0_hat = x0_hat.clamp(-1, 1)
            x = ab_p.sqrt() * x0_hat + (1-ab_p).sqrt() * pred_n
        return x.clamp(-1, 1)


if __name__ == "__main__":
    unet   = UNet(base_ch=64)
    model  = GaussianDiffusion(unet, T=1000)
    x      = torch.randn(2, 3, 256, 256)
    loss   = model.p_loss(x)
    sample = model.ddim_sample((2, 3, 256, 256), device="cpu", n_steps=10)
    params = sum(p.numel() for p in unet.parameters()) / 1e6
    print(f"U-Net params: {params:.2f}M")
    print(f"Training loss: {loss.item():.4f}")
    print(f"Sample shape: {sample.shape}")
