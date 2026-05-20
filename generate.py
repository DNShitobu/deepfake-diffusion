"""DDPM — Inference Script (DDPM + DDIM samplers)"""
import argparse, torch
from torchvision.utils import save_image
from model import UNet, GaussianDiffusion

def load(ckpt, base_ch=64, T=1000, use_ema=True, device="cpu"):
    unet = UNet(base_ch=base_ch).to(device)
    diffusion = GaussianDiffusion(unet, T=T).to(device)
    state = torch.load(ckpt, map_location=device)
    weights = state.get("ema" if use_ema else "model", state.get("model"))
    if use_ema and "ema" in state:
        unet.load_state_dict({k: v.to(device) for k, v in state["ema"].items()})
    else:
        unet.load_state_dict(state["model"])
    diffusion.eval()
    return diffusion

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",     required=True)
    p.add_argument("--n",        type=int, default=16)
    p.add_argument("--sampler",  default="ddim", choices=["ddpm","ddim"])
    p.add_argument("--steps",    type=int, default=50, help="DDIM steps (ignored for DDPM)")
    p.add_argument("--out",      default="generated_faces.png")
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--base_ch",  type=int, default=64)
    p.add_argument("--T",        type=int, default=1000)
    p.add_argument("--seed",     type=int, default=42)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    diffusion = load(args.ckpt, args.base_ch, args.T, device=device)
    shape = (args.n, 3, args.img_size, args.img_size)
    with torch.no_grad():
        if args.sampler == "ddim":
            print(f"DDIM sampling ({args.steps} steps)...")
            imgs = diffusion.ddim_sample(shape, device, n_steps=args.steps)
        else:
            print(f"DDPM sampling ({args.T} steps) — this will take a while...")
            imgs = diffusion.ddpm_sample(shape, device)
    save_image(imgs * 0.5 + 0.5, args.out, nrow=4)
    print(f"Saved {args.n} faces to {args.out}")
