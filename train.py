import torch
import torch.nn as nn
import torch.nn.functional as F
from fastai.vision.all import *
from transformers import CLIPTextModel, CLIPTokenizer
from models import VAE, FlowDenoiser
from pathlib import Path
import json
from PIL import Image
import numpy as np
from torchvision import transforms

# Config
CONFIG = {
    "img_size": 256,
    "latent_size": 16,
    "latent_channels": 16,
    "batch_size": 2, # Smaller batch for safety
    "lr": 1e-4,
    "epochs": 15,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

class FlowModel(nn.Module):
    def __init__(self, vae, denoiser, clip_text_model, tokenizer):
        super().__init__()
        self.vae = vae
        self.denoiser = denoiser
        self.clip = clip_text_model
        self.tokenizer = tokenizer

        # Freeze CLIP
        for param in self.clip.parameters():
            param.requires_grad = False

    def forward(self, x, captions):
        # x: [b, 3, 256, 256]
        device = x.device
        b = x.shape[0]

        # 1. VAE encoding
        rec, z_data = self.vae(x)

        # 2. Get CLIP embeddings
        inputs = self.tokenizer(captions, padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(device)
        context = self.clip(**inputs).last_hidden_state

        # 3. Flow Matching
        t = torch.rand(b, device=device)
        z_noise = torch.randn_like(z_data)

        # x_t = (1-t)z_noise + t*z_data
        t_view = t.view(-1, 1, 1, 1)
        z_t = (1 - t_view) * z_noise + t_view * z_data

        # Predict velocity
        v_pred = self.denoiser(z_t, t, context)

        # Target velocity is (z_data - z_noise)
        v_target = z_data - z_noise

        return rec, v_pred, v_target

def flow_loss(outs, x):
    rec, v_pred, v_target = outs

    # Reconstruction loss
    loss_rec = F.mse_loss(rec, x)

    # Flow matching loss
    loss_flow = F.mse_loss(v_pred, v_target)

    return loss_rec + loss_flow

class SnapshotCallback(Callback):
    def __init__(self, prompts, vae, denoiser, clip, tokenizer, folder="snapshots"):
        self.prompts = prompts
        self.vae = vae
        self.denoiser = denoiser
        self.clip = clip
        self.tokenizer = tokenizer
        self.folder = Path(folder)
        self.folder.mkdir(exist_ok=True)

    def after_epoch(self):
        self.vae.eval()
        self.denoiser.eval()
        device = next(self.vae.parameters()).device

        with torch.no_grad():
            for i, prompt in enumerate(self.prompts):
                # Encode prompt
                inputs = self.tokenizer([prompt], padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(device)
                context = self.clip(**inputs).last_hidden_state

                # Sample
                z = torch.randn(1, CONFIG["latent_channels"], CONFIG["latent_size"], CONFIG["latent_size"]).to(device)
                dt = 0.05
                for t_val in torch.linspace(0, 1, 20):
                    t = torch.tensor([t_val], device=device)
                    v = self.denoiser(z, t, context)
                    z = z + v * dt

                # Decode
                img_t = self.vae.decode(z)
                img_t = (img_t.clamp(-1, 1) + 1) / 2
                img_pil = Image.fromarray((img_t[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
                img_pil.save(self.folder / f"epoch_{self.epoch}_{i}.png")

        self.vae.train()
        self.denoiser.train()

class ModelSaveCallback(Callback):
    def after_epoch(self):
        torch.save(self.model.vae.state_dict(), f"vae_epoch_{self.epoch}.pth")
        torch.save(self.model.denoiser.state_dict(), f"denoiser_epoch_{self.epoch}.pth")

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, items, size):
        self.items = items
        self.size = size
        self.tfms = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        img_path, caption = self.items[idx]
        img = Image.open(img_path).convert("RGB")
        img_t = self.tfms(img)
        return img_t, caption, img_t

def get_dls():
    data_dir = Path("Data")
    metadata_file = data_dir / "metadata.jsonl"

    with open(metadata_file, "r") as f:
        items = [json.loads(line) for line in f]

    # Simple list of (image_path, caption)
    data = [(data_dir / "images" / item["file_name"], item["text"]) for item in items]

    ds = CustomDataset(data, CONFIG["img_size"])
    return DataLoaders.from_dsets(ds, ds, bs=CONFIG["batch_size"], device=CONFIG["device"], n_inp=2, num_workers=0)

def train():
    # 1. Prep Data
    import dataset
    dataset.prepare_data(limit=500)

    dls = get_dls()

    # 2. Load Models
    print("Loading CLIP...")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(CONFIG["device"])

    vae = VAE(latent_channels=CONFIG["latent_channels"]).to(CONFIG["device"])
    denoiser = FlowDenoiser(
        latent_channels=CONFIG["latent_channels"],
        context_dim=clip_model.config.hidden_size
    ).to(CONFIG["device"])

    model = FlowModel(vae, denoiser, clip_model, tokenizer)

    # 3. Learner
    prompts = ["a beautiful sunset", "a cute cat", "a futuristic city", "a mountain landscape"]
    cbs = [
        SnapshotCallback(prompts, vae, denoiser, clip_model, tokenizer),
        ModelSaveCallback()
    ]

    learn = Learner(dls, model, loss_func=flow_loss, cbs=cbs)

    print("Starting training...")
    learn.fit_one_cycle(CONFIG["epochs"], CONFIG["lr"])

if __name__ == "__main__":
    train()
