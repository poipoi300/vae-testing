import torch
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from shared_config import CONFIG

def calculate_collapse_score(img_tensor):
    """
    Calculates per-pixel channel similarity.
    Lower is better (means more color diversity).
    img_tensor: [C, H, W] in range [0, 1]
    """
    # Calculate pairwise squared differences between channels
    diff_rg = (img_tensor[0] - img_tensor[1])**2
    diff_gb = (img_tensor[1] - img_tensor[2])**2
    diff_br = (img_tensor[2] - img_tensor[0])**2
    
    # Minimum squared difference (nearest neighbor in channel space)
    min_diff_sq = torch.min(torch.stack([diff_rg, diff_gb, diff_br]), dim=0)[0]
    
    # Score: exp(-min_diff_sq / 2.0). 
    # High score (~1.0) means channels are very close (collapse).
    # Low score means channels are distinct.
    score = torch.exp(-min_diff_sq / 2.0).mean().item()
    return score

def main():
    snapshot_dir = Path(CONFIG["snapshots_dir"])
    if not snapshot_dir.exists():
        print(f"Snapshot directory {snapshot_dir} not found.")
        return

    # Find all VAE input/output pairs
    # Naming: epoch_n_vae_inp.png, epoch_n_vae_out.png
    inputs = sorted(list(snapshot_dir.glob("*_vae_inp.png")))
    
    results = []
    to_tensor = transforms.ToTensor()

    print(f"Analyzing {len(inputs)} VAE snapshot pairs...")
    
    for inp_path in inputs:
        out_path = Path(str(inp_path).replace("_vae_inp.png", "_vae_out.png"))
        
        if not out_path.exists():
            continue
            
        epoch = inp_path.stem.split("_")[1]
        
        inp_img = to_tensor(Image.open(inp_path).convert("RGB"))
        out_img = to_tensor(Image.open(out_path).convert("RGB"))
        
        inp_score = calculate_collapse_score(inp_img)
        out_score = calculate_collapse_score(out_img)
        
        results.append({
            "epoch": int(epoch),
            "input_file": inp_path.name,
            "output_file": out_path.name,
            "input_collapse_score": inp_score,
            "output_collapse_score": out_score,
            "collapse_delta": out_score - inp_score
        })

    if not results:
        print("No VAE snapshot pairs found.")
        return

    df = pd.DataFrame(results).sort_values("epoch")
    output_csv = "vae_collapse_analysis.csv"
    df.to_csv(output_csv, index=False)
    
    print(f"\nAnalysis complete. Results saved to {output_csv}")
    print("\nSummary (Averages):")
    print(df[["input_collapse_score", "output_collapse_score", "collapse_delta"]].mean())
    
    print("\nDetailed View:")
    print(df[["epoch", "input_collapse_score", "output_collapse_score", "collapse_delta"]].to_string(index=False))

if __name__ == "__main__":
    main()
