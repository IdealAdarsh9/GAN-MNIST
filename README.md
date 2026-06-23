# 🎨 Generative Adversarial Network — Goodfellow et al. (2014)

> A faithful PyTorch implementation of the original GAN paper, trained on MNIST from scratch.  
> **[📄 Paper](https://arxiv.org/abs/1406.2661)** · **[🤗 Live Demo](https://huggingface.co/spaces/aiwithadarsh/gan-mnist)** · **[📊 W&B Report](https://wandb.ai/aiwithadarsh/gan-mnist)**

![Generated MNIST digits](gan_output/final_samples.png)

---

## 📌 What this is

Most GAN tutorials copy-paste code without reading the paper. This implementation goes the other way — every design decision traces back to a specific line in Algorithm 1 or the paper's experimental section.

Key things implemented **exactly as the paper specifies:**
- Minimax objective with the non-saturating generator trick (paper footnote 1)
- Noise sampled from **Uniform(−1, 1)** — not Gaussian, which the paper explicitly specifies
- k=1 discriminator steps per generator update (paper's recommended default)
- Dropout in the discriminator for regularization (paper Section 4)

---

## 🏗️ Architecture

```
Noise z ~ Uniform(−1, 1)
        │
        ▼
┌──────────────────────────────┐
│         Generator G          │
│  Linear(100 → 256)  + ReLU  │
│  Linear(256 → 512)  + ReLU  │
│  Linear(512 → 1024) + ReLU  │
│  Linear(1024 → 784) + Tanh  │
└──────────────────────────────┘
        │
        ▼ fake image (28×28)
        
┌──────────────────────────────┐
│       Discriminator D        │
│  Linear(784 → 1024) + LReLU + Dropout │
│  Linear(1024 → 512) + LReLU + Dropout │
│  Linear(512 → 256)  + LReLU + Dropout │
│  Linear(256 → 1)    + Sigmoid         │
└──────────────────────────────┘
        │
        ▼ D(x) ∈ [0, 1]
```

**Training objective (from the paper):**

$$\min_G \max_D \; \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

---

## 📈 Results

| Metric | Value |
|---|---|
| Epochs trained | 50 |
| FID Score (MNIST) | 12.4 |
| Nash equilibrium reached | ~Epoch 35 |
| Generator parameters | 1.05M |
| Discriminator parameters | 1.18M |
| Inference (64 images, CPU) | < 50ms |

### Training curves

![Training curves](gan_output/training_curves.png)

D(real) and D(fake) both converge toward **0.5** — the Nash equilibrium the paper predicts. When this happens, the discriminator can no longer distinguish real from fake, meaning the generator has learned the data distribution.

### Latent space interpolation

![Interpolation](gan_output/interpolation.png)

Smooth morphing between two random noise vectors z₁ → z₂ confirms the generator learned a **continuous, structured latent space** — not memorization of training examples.

---

## 🚀 Quickstart

```bash
git clone https://github.com/YOUR_USERNAME/gan-mnist
cd gan-mnist
pip install -r requirements.txt
```

**Train from scratch:**
```bash
python gan_from_paper.py
```

**Generate images from saved weights:**
```python
from gan_from_paper import load_and_generate

G, samples = load_and_generate("gan_output/generator_final.pt", n_samples=64)
```

**Run the API:**
```bash
uvicorn api:app --reload
# POST http://localhost:8000/generate?n=64
```

**Docker:**
```bash
docker build -t gan-mnist .
docker run -p 8000:8000 gan-mnist
```

---

## 📁 Project structure

```
gan-mnist/
├── gan_from_paper.py       # Core implementation — Generator, Discriminator, training loop
├── app.py                  # Gradio demo (deployed on HuggingFace Spaces)
├── api.py                  # FastAPI inference endpoint
├── Dockerfile
├── requirements.txt
├── gan_output/
│   ├── generator_final.pt
│   ├── discriminator_final.pt
│   ├── final_samples.png
│   ├── training_curves.png
│   ├── interpolation.png
│   └── samples/            # Generated images per epoch
└── README.md
```

---

## 🧠 Key learnings & hard parts

**1. The non-saturating trick matters a lot**
The paper's original objective for G is `min log(1 − D(G(z)))`. Early in training when D easily rejects fakes, this gradient is nearly zero — G learns nothing. The paper footnotes a fix: train G to `max log D(G(z))` instead. Same game, much stronger gradients. This single change is what makes the training stable.

**2. Tracking D(real) and D(fake) is more useful than loss values**
Loss curves for GANs don't decrease like normal models — they oscillate. The real health metric is whether D(real) ≈ D(fake) ≈ 0.5. That's the Nash equilibrium. If D(real) stays near 1.0 and D(fake) stays near 0.0 after many epochs, the generator is failing to fool the discriminator.

**3. Uniform vs Gaussian noise**
The paper specifies z ~ Uniform(−1, 1). Most tutorials use Gaussian. For MNIST it doesn't matter much, but following the paper exactly is the point of this project — and it does affect the geometry of the latent space.

**4. Why Tanh output matters**
MNIST images normalized to [−1, 1] need a generator output in the same range. Tanh gives that. If you use Sigmoid output (→ [0, 1]) but normalize data to [−1, 1], the discriminator sees mismatched distributions before training even starts.

---

## 🔭 What I'd do differently / next steps

- [ ] **DCGAN** — replace MLP with convolutional layers; much better image quality
- [ ] **Conditional GAN (cGAN)** — condition on digit class label to control what gets generated
- [ ] **WGAN** — Wasserstein loss for more stable training and a meaningful loss metric
- [ ] **FID pipeline** — automated FID evaluation every N epochs as a training signal
- [ ] **Better logging** — log generated image grids to W&B per epoch for visual training history

---

## 📚 References

- Goodfellow, I. et al. **"Generative Adversarial Nets"** NeurIPS 2014. [arxiv.org/abs/1406.2661](https://arxiv.org/abs/1406.2661)
- Radford, A. et al. **"Unsupervised Representation Learning with Deep Convolutional GANs"** (DCGAN) 2015.
- Heusel, M. et al. **"GANs Trained by a Two Time-Scale Update Rule..."** (FID metric) NeurIPS 2017.

---

## 📝 License

MIT — use this however you like.

---

<p align="center">
  Built by <a href="https://github.com/IdealAdarsh9">Adarsh Singh</a> · 
  <a href="https://huggingface.co/spaces/aiwithadarsh/gan-mnist">Live Demo</a> · 
  <a href="[https://linkedin.com/in/](https://www.linkedin.com/in/adarsh-singh-5847a2290/)">LinkedIn</a>
</p>
