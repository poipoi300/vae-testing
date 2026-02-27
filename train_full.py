import os
from pathlib import Path
from shared_config import CONFIG
from train_vae import train_vae
from train_denoiser import train_denoiser

def main():
    checkpoints_dir = Path(CONFIG["checkpoints_dir"])
    checkpoints_dir.mkdir(exist_ok=True)
    
    vae_checkpoint = checkpoints_dir / "vae_best.pth"
    denoiser_checkpoint = checkpoints_dir / "denoiser_best.pth"
    
    # 1. VAE Stage
    if not vae_checkpoint.exists():
        print("--- VAE checkpoint not found. Starting VAE training... ---")
        train_vae(epochs=CONFIG["vae_epochs"])
    else:
        print("--- VAE checkpoint found. Loading... ---")
        train_vae(epochs=0)

    # 2. Denoiser Stage
    if not denoiser_checkpoint.exists():
        print("--- Denoiser checkpoint not found. Starting Denoiser training... ---")
        train_denoiser(epochs=CONFIG["denoiser_epochs"])
    else:
        print("--- Denoiser checkpoint found. Loading... ---")
        train_denoiser(epochs=0)

    print("--- Training process complete! ---")

if __name__ == "__main__":
    main()
