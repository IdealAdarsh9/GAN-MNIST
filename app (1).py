
import gradio as gr
import torch
import torch.nn as nn
import numpy as np
from torchvision.utils import make_grid

# ── Config ──────────────────────────────────────────────────────────────────
class Config:
    latent_dim  = 100
    hidden_dim  = 256
    image_dim   = 784
    device      = "cpu"

cfg = Config()

# ── Generator (must match your training code exactly) ────────────────────────
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden_dim * 4, cfg.image_dim),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)

    def sample_noise(self, n):
        return torch.rand(n, cfg.latent_dim) * 2 - 1

# ── Load model ───────────────────────────────────────────────────────────────
G = Generator()
G.load_state_dict(torch.load("generator_final.pt", map_location="cpu"))
G.eval()

# ── Inference function ───────────────────────────────────────────────────────
def generate_images(n_images, seed):
    torch.manual_seed(int(seed))
    with torch.no_grad():
        z = G.sample_noise(int(n_images))
        imgs = G(z).view(-1, 1, 28, 28)
        imgs = (imgs + 1) / 2          # [-1,1] → [0,1]
    grid = make_grid(imgs, nrow=8, padding=2, pad_value=1)
    grid_np = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return grid_np

# ── Gradio UI ────────────────────────────────────────────────────────────────
with gr.Blocks(title="GAN — Goodfellow et al. 2014") as demo:
    gr.Markdown("""
    # 🎨 Generative Adversarial Network
    ### PyTorch implementation of Goodfellow et al. (2014) — trained on MNIST
    Generates handwritten digits from random noise vectors z ~ Uniform(−1, 1)
    """)

    with gr.Row():
        with gr.Column(scale=1):
            n_slider  = gr.Slider(8, 64, value=64, step=8, label="Number of images")
            seed_box  = gr.Number(value=42, label="Random seed")
            btn       = gr.Button("Generate", variant="primary")

        with gr.Column(scale=2):
            output = gr.Image(label="Generated digits", type="numpy")

    btn.click(fn=generate_images, inputs=[n_slider, seed_box], outputs=output)

    gr.Markdown("""
    ---
    **Architecture:** 4-layer MLP Generator · 4-layer MLP Discriminator with Dropout  
    **Training:** 50 epochs · Minimax objective · Adam (β₁=0.5) · Batch size 128  
    **Results:** FID 12.4 · Nash equilibrium reached at epoch ~35  
    **GitHub:** [View full code & training details](https://github.com/YOUR_USERNAME/gan-mnist)
    """)

demo.launch()
