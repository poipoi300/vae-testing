import torch
import torch.nn as nn
import torch.nn.functional as F
from fastai.vision.all import *
from transformers import CLIPTextModel, CLIPTokenizer
from models import VAE, FlowDenoiser
from shared_config import CONFIG
from utils import make_path_safe
from pathlib import Path
import json
from PIL import Image
import numpy as np
from torchvision import transforms

class FlowModelWrapper(nn.Module):
    def __init__(self, vae, denoiser, clip, tokenizer):
        super().__init__()
        self.vae = vae
        self.denoiser = denoiser
        self.clip = clip
        self.tokenizer = tokenizer
        
        # Freeze VAE and CLIP
        for param in self.vae.parameters(): param.requires_grad = False
        for param in self.clip.parameters(): param.requires_grad = False

    def forward(self, x, captions):
        device = x.device
        b = x.shape[0]
        
        with torch.no_grad():
            _, z_data = self.vae(x)
            inputs = self.tokenizer(captions, padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(device)
            context = self.clip(**inputs).last_hidden_state

        t = torch.rand(b, device=device)
        z_noise = torch.randn_like(z_data)
        t_view = t.view(-1, 1, 1, 1)
        z_t = (1 - t_view) * z_noise + t_view * z_data
        
        v_pred = self.denoiser(z_t, t, context)
        v_target = z_data - z_noise
        return v_pred, v_target

def flow_loss(outs, target_unused):
    v_pred, v_target = outs
    return F.mse_loss(v_pred, v_target)

class Denormalize(object):
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)
    def __call__(self, tensor):
        return tensor * self.std.to(tensor.device) + self.mean.to(tensor.device)

class DenoiserSnapshotCallback(Callback):
    def __init__(self, prompts, folder=CONFIG["snapshots_dir"]):
        self.prompts = prompts
        self.folder = Path(folder)
        self.folder.mkdir(exist_ok=True)
        self.denorm = Denormalize(CONFIG["dataset_mean"], CONFIG["dataset_std"])

    def after_epoch(self):
        self.model.eval()
        device = next(self.model.parameters()).device
        
        with torch.no_grad():
            for prompt in self.prompts:
                # Encode prompt
                inputs = self.model.tokenizer([prompt], padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(device)
                context = self.model.clip(**inputs).last_hidden_state

                # Sample
                z = torch.randn(1, CONFIG["latent_channels"], CONFIG["latent_size"], CONFIG["latent_size"]).to(device)
                dt = 0.05
                for t_val in torch.linspace(0, 1, 20):
                    t = torch.tensor([t_val], device=device)
                    v = self.model.denoiser(z, t, context)
                    z = z + v * dt

                # Decode
                img_t = self.model.vae.decode(z)
                img_t = self.denorm(img_t[0]).clamp(0, 1)
                img_pil = Image.fromarray((img_t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
                
                safe_prompt = make_path_safe(prompt)
                img_pil.save(self.folder / f"epoch_{self.epoch}_{safe_prompt}.png")

        self.model.train()

class DenoiserDataset(torch.utils.data.Dataset):
    def __init__(self, items, size):
        self.items = items
        self.tfms = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(CONFIG["dataset_mean"], CONFIG["dataset_std"])
        ])
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        img_path, caption = self.items[idx]
        img = Image.open(img_path).convert("RGB")
        img_t = self.tfms(img)
        return img_t, caption, img_t # image, caption as inputs, image as target (unused but needed by fastai)

def get_denoiser_dls():
    data_dir = Path("Data")
    metadata_file = data_dir / "metadata.jsonl"
    with open(metadata_file, "r") as f:
        items = [json.loads(line) for line in f]
    data = [(data_dir / "images" / item["file_name"], item["text"]) for item in items]
    
    n = len(data)
    train_n = int(0.9 * n)
    train_ds = DenoiserDataset(data[:train_n], CONFIG["img_size"])
    valid_ds = DenoiserDataset(data[train_n:], CONFIG["img_size"])
    
    return DataLoaders.from_dsets(train_ds, valid_ds, bs=CONFIG["batch_size"], device=CONFIG["device"], num_workers=0, n_inp=2)

def train_denoiser(epochs=None):
    if epochs is None: epochs = CONFIG["denoiser_epochs"]
    
    import dataset
    dataset.prepare_data(limit=CONFIG["dataset_limit"])
    
    dls = get_denoiser_dls()
    
    vae = VAE(latent_channels=CONFIG["latent_channels"]).to(CONFIG["device"])
    # Load VAE weights from checkpoints/vae_best.pth
    vae_path = Path(CONFIG["checkpoints_dir"]) / "vae_best.pth"
    if not vae_path.exists():
        print("VAE checkpoint not found! Please train VAE first.")
        return
    
    print(f"Loading VAE from {vae_path}...")
    state_dict = torch.load(vae_path, map_location=CONFIG["device"], weights_only=False)
    try:
        if 'model' in state_dict:
            # If saved via SaveModelCallback, it's wrapped in VAEWrapper
            vae_state = {k.replace('vae.', ''): v for k, v in state_dict['model'].items() if k.startswith('vae.')}
            vae.load_state_dict(vae_state)
        else:
            vae.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"Error: Could not load VAE weights due to architecture mismatch: {e}")
        print("Please delete the old VAE checkpoint and re-run VAE training.")
        return

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(CONFIG["device"])
    
    denoiser = FlowDenoiser(
        latent_channels=CONFIG["latent_channels"],
        context_dim=clip_model.config.hidden_size
    ).to(CONFIG["device"])
    
    model = FlowModelWrapper(vae, denoiser, clip_model, tokenizer)
    
    checkpoints_path = Path(CONFIG["checkpoints_dir"])
    checkpoints_path.mkdir(exist_ok=True)
    prompts = ["a photo of a cat", "a photo of a dog", "a beautiful sunset", "a futuristic city"]
    
    cbs = [
        DenoiserSnapshotCallback(prompts),
        SaveModelCallback(monitor='valid_loss', fname='denoiser_best', every_epoch=False, with_opt=False, reset_on_fit=True)
    ]
    
    learn = Learner(dls, model, loss_func=flow_loss, cbs=cbs, path=checkpoints_path, model_dir='.')
    
    if (checkpoints_path / "denoiser_best.pth").exists():
        print("Loading existing Denoiser checkpoint...")
        learn.load('denoiser_best')
        
    if epochs > 0:
        print(f"Training Denoiser for {epochs} epochs...")
        learn.fit_one_cycle(epochs, CONFIG["denoiser_lr"])
    else:
        print("Skipping Denoiser training (epochs=0).")

if __name__ == "__main__":
    train_denoiser()
