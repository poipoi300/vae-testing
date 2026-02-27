import torch

CONFIG = {
    "img_size": 256,
    "latent_size": 32,
    "latent_channels": 12,
    "batch_size": 32,
    "vae_lr": 2e-5,
    "denoiser_lr": 2e-5,
    "vae_epochs": 20,
    "denoiser_epochs": 20,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dataset_limit": 5000,
    "checkpoints_dir": "checkpoints",
    "snapshots_dir": "snapshots",
    "dataset_mean": [0.470, 0.429, 0.375],
    "dataset_std": [0.225, 0.222, 0.226]
}
