"""
DDPM Face Generation — Training Script
========================================
Trains the diffusion model on CelebA/FFHQ.

Training tips:
  - Use mixed-precision (fp16) for ~2x speedup
  - EMA of model weights produces much cleaner samples
  - FID should drop steadily; monitor every 10 epochs
  - Expected training time: ~24h on single A100 for 256x256 CelebA
"""

import argparse, time
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.datasets as dsets
from torchvision.utils import save_image
from torch.cuda.amp import GradScaler, autocast

from model import UNet, GaussianDiffusion


# ── EMA ───────────────────────────────────────────────────────────────────────

class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay  = decay
        self.shadow = {k: v.clone().float().cpu() for k, v in model.state_dict().items()}
    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k] = self.shadow[k] * self.decay + v.float().cpu() * (1 - self.decay)
    def copy_to(self, model):
        d = model.device if hasattr(model, "device") else next(model.parameters()).device
        model.load_state_dict({k: v.to(d) for k, v in self.shadow.items()})


# ── Dataset ────────────────────────────────────────────────────────────────────

def get_loader(root, batch_size, size, dataset):
    tf = T.Compose([
        T.CenterCrop(148) if dataset == "celeba" else T.Resize((size, size)),
        T.Resize((size, size)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3),
    ])
    if dataset == "celeba":
        ds = dsets.CelebA(root, split="train", transform=tf, download=True)
    else:
        ds = dsets.ImageFolder(root, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=4, pin_memory=True, drop_last=True)


# ── Trainer ────────────────────────────────────────────────────────────────────

class DDPMTrainer:
    def __init__(self, args):
        self.args   = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        unet         = UNet(base_ch=args.base_ch).to(self.device)
        self.diffusion = GaussianDiffusion(unet, T=args.T).to(self.device)
        self.ema       = EMA(unet)
        self.opt       = optim.AdamW(unet.parameters(), lr=args.lr, weight_decay=0.01)
        self.sch       = optim.lr_scheduler.CosineAnnealingLR(self.opt, args.epochs)
        self.scaler    = GradScaler(enabled=args.fp16)
        self.out_dir   = Path(args.out_dir); self.out_dir.mkdir(parents=True, exist_ok=True)

    def train_step(self, x0):
        x0 = x0.to(self.device)
        self.opt.zero_grad()
        with autocast(enabled=self.args.fp16):
            loss = self.diffusion.p_loss(x0)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        torch.nn.utils.clip_grad_norm_(self.diffusion.model.parameters(), 1.0)
        self.scaler.step(self.opt)
        self.scaler.update()
        self.ema.update(self.diffusion.model)
        return loss.item()

    def run(self, loader):
        for epoch in range(1, self.args.epochs + 1):
            t0 = time.time()
            losses = []
            for batch in loader:
                imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
                losses.append(self.train_step(imgs))
            elapsed = time.time() - t0
            print(f"[Epoch {epoch:03d}/{self.args.epochs}] "
                  f"Loss={np.mean(losses):.6f} LR={self.opt.param_groups[0]['lr']:.2e} "
                  f"({elapsed:.0f}s)")
            if epoch % self.args.save_every == 0:
                print(f"  Generating DDIM samples (50 steps)...")
                # Temporarily copy EMA weights
                self.ema.copy_to(self.diffusion.model)
                self.diffusion.eval()
                samples = self.diffusion.ddim_sample(
                    (16, 3, self.args.img_size, self.args.img_size),
                    device=self.device, n_steps=50
                )
                self.diffusion.train()
                save_image(samples * 0.5 + 0.5, self.out_dir / f"samples_epoch_{epoch:04d}.png", nrow=4)
                torch.save({
                    "epoch": epoch,
                    "model": self.diffusion.model.state_dict(),
                    "ema":   self.ema.shadow,
                    "opt":   self.opt.state_dict(),
                }, self.out_dir / f"ckpt_epoch_{epoch:04d}.pt")
            self.sch.step()


def parse_args():
    p = argparse.ArgumentParser("DDPM Trainer")
    p.add_argument("--data",       default="./data/celeba")
    p.add_argument("--dataset",    default="celeba", choices=["celeba","ffhq","custom"])
    p.add_argument("--out_dir",    default="./outputs")
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--batch_size", type=int,   default=8)
    p.add_argument("--lr",         type=float, default=2e-4)
    p.add_argument("--base_ch",    type=int,   default=64)
    p.add_argument("--img_size",   type=int,   default=256)
    p.add_argument("--T",          type=int,   default=1000)
    p.add_argument("--save_every", type=int,   default=5)
    p.add_argument("--fp16",       action="store_true", help="Use mixed precision training")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    loader = get_loader(args.data, args.batch_size, args.img_size, args.dataset)
    DDPMTrainer(args).run(loader)
