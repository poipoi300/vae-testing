import torch
import torch.nn as nn
import torch.nn.functional as F
from fastai.vision.all import *
from models import VAE
from shared_config import CONFIG
from utils import make_path_safe
from pathlib import Path
import json
from PIL import Image
import numpy as np
from torchvision import transforms
import random

class VAEWrapper(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, x):
        rec, z = self.vae(x)
        return rec

def get_collapse(img):
    # img: [b, 3, h, w]
    diff_rg = (img[:, 0] - img[:, 1])**2
    diff_gb = (img[:, 1] - img[:, 2])**2
    diff_br = (img[:, 2] - img[:, 0])**2
    min_diff_sq = torch.min(torch.stack([diff_rg, diff_gb, diff_br], dim=-1), dim=-1)[0]
    return torch.exp(-min_diff_sq / 2.0)

def vae_loss(pred, target, model=None):
    mse = F.mse_loss(pred, target)
    l1 = F.l1_loss(pred, target)
    psnr_term = torch.log10(mse + 1e-8)
    
    base_loss = mse + l1 + 0.05 * psnr_term
    
    # Color Similarity Divergence Penalty
    collapse_pred = get_collapse(pred)
    collapse_target = get_collapse(target)
    
    # Penalize divergence from target's collapse level
    divergence = torch.abs(collapse_pred - collapse_target)
    
    # Exponential penalty for divergence
    similarity_penalty = (torch.exp(divergence * 3.0) - 1).mean()
    
    # Constant weight for the flat penalty
    weight = 0.1 
    
    return base_loss + weight * similarity_penalty

# EpochTrackerCallback removed as it's no longer needed for decay

class Denormalize(object):
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)
    def __call__(self, tensor):
        return tensor * self.std.to(tensor.device) + self.mean.to(tensor.device)

class VAESnapshotCallback(Callback):
    def __init__(self, folder=CONFIG["snapshots_dir"]):
        self.folder = Path(folder)
        self.folder.mkdir(exist_ok=True)
        self.denorm = Denormalize(CONFIG["dataset_mean"], CONFIG["dataset_std"])

    def after_epoch(self):
        self.model.eval()
        # Grab a batch from validation set
        try:
            xb, yb = self.learn.dls.valid.one_batch()
            xb = xb.to(CONFIG["device"])
        except:
            return

        with torch.no_grad():
            rec = self.model(xb)
            
            # Save first item in batch
            inp = self.denorm(xb[0]).clamp(0, 1)
            out = self.denorm(rec[0]).clamp(0, 1)
            
            inp_pil = Image.fromarray((inp.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            out_pil = Image.fromarray((out.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            
            inp_pil.save(self.folder / f"epoch_{self.epoch}_vae_inp.png")
            out_pil.save(self.folder / f"epoch_{self.epoch}_vae_out.png")
        
        self.model.train()

class ChannelNoise(object):
    def __init__(self, std=0.05):
        self.std = std
    def __call__(self, tensor):
        # tensor is [C, H, W]
        # Adding independent noise to each channel
        noise = torch.randn_like(tensor) * self.std
        return tensor + noise

class VAEDataset(torch.utils.data.Dataset):
    def __init__(self, items, size, is_train=False, noise_prob=0.05):
        self.items = items
        self.size = size
        self.noise_prob = noise_prob if is_train else 0
        if is_train:
            self.tfms = transforms.Compose([
                transforms.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.8, 1.2)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(CONFIG["dataset_mean"], CONFIG["dataset_std"]),
                ChannelNoise(std=0.02)
            ])
        else:
            self.tfms = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(CONFIG["dataset_mean"], CONFIG["dataset_std"])
            ])
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        if self.noise_prob > 0 and random.random() < self.noise_prob:
            # Return pure noise image with independent channels
            noise = torch.rand(3, self.size, self.size)
            norm_noise = transforms.Normalize(CONFIG["dataset_mean"], CONFIG["dataset_std"])(noise)
            return norm_noise, norm_noise

        img_path, _ = self.items[idx]
        img = Image.open(img_path).convert("RGB")
        img_t = self.tfms(img)
        return img_t, img_t

def get_vae_dls():
    data_dir = Path("Data")
    metadata_file = data_dir / "metadata.jsonl"
    with open(metadata_file, "r") as f:
        items = [json.loads(line) for line in f]
    data = [(data_dir / "images" / item["file_name"], item["text"]) for item in items]
    
    # Split data
    n = len(data)
    train_n = int(0.9 * n)
    train_ds = VAEDataset(data[:train_n], CONFIG["img_size"], is_train=True)
    valid_ds = VAEDataset(data[train_n:], CONFIG["img_size"], is_train=False)
    
    return DataLoaders.from_dsets(train_ds, valid_ds, bs=CONFIG["batch_size"], device=CONFIG["device"], num_workers=0)

def train_vae(epochs=None):
    if epochs is None: epochs = CONFIG["vae_epochs"]
    
    import dataset
    dataset.prepare_data(limit=CONFIG["dataset_limit"])
    
    dls = get_vae_dls()
    vae = VAE(latent_channels=CONFIG["latent_channels"]).to(CONFIG["device"])
    model = VAEWrapper(vae)
    
    checkpoints_path = Path(CONFIG["checkpoints_dir"])
    checkpoints_path.mkdir(exist_ok=True)
    
    cbs = [
        VAESnapshotCallback(),
        SaveModelCallback(monitor='valid_loss', fname='vae_best', every_epoch=False, with_opt=False, reset_on_fit=True)
    ]
    
    # Wrap loss to pass model
    def loss_wrapper(p, t): return vae_loss(p, t, model=model)
    
    learn = Learner(dls, model, loss_func=loss_wrapper, cbs=cbs, path=checkpoints_path, model_dir='.')
    
    if (checkpoints_path / "vae_best.pth").exists():
        print("Loading existing VAE checkpoint...")
        try:
            learn.load('vae_best')
        except RuntimeError as e:
            print(f"Could not load VAE checkpoint (likely architecture mismatch): {e}")
            print("Starting VAE training from scratch.")
        
    if epochs > 0:
        print(f"Training VAE for {epochs} epochs...")
        learn.fit_one_cycle(epochs, CONFIG["vae_lr"])
    else:
        print("Skipping VAE training (epochs=0).")
    return vae

if __name__ == "__main__":
    train_vae()
