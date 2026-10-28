"""Model wrappers. Architectures are copied from the research code (src/cwgangp_train.py) and the third-party classifier
(external_models/brain_tumor_classifier/src/model.py, Apache-2.0); weights are loaded with weights_only=True and strict=True.
A model that fails to load is reported as unavailable; nothing is substituted."""
import base64, hashlib, io, threading
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from . import config

Z_DIM, EMB_DIM, NUM_CLASSES = 100, 50, 4


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class CondGenerator(nn.Module):
    """Same definition as CondGenerator in src/cwgangp_train.py."""

    def __init__(self, z_dim: int = Z_DIM, emb_dim: int = EMB_DIM, ch: int = 64):
        super().__init__()
        self.embed = nn.Embedding(NUM_CLASSES, emb_dim)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim + emb_dim, ch * 8, 4, 1, 0, bias=False), nn.BatchNorm2d(ch * 8), nn.ReLU(True),
            nn.ConvTranspose2d(ch * 8, ch * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ch * 4), nn.ReLU(True),
            nn.ConvTranspose2d(ch * 4, ch * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ch * 2), nn.ReLU(True),
            nn.ConvTranspose2d(ch * 2, ch, 4, 2, 1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(True),
            nn.ConvTranspose2d(ch, 1, 4, 2, 1, bias=False), nn.Tanh(),
        )

    def forward(self, z, y):
        h = torch.cat([z, self.embed(y)], dim=1)
        return self.net(h.view(h.size(0), -1, 1, 1))


class BrainTumorEfficientNet(nn.Module):
    """Architecture of the third-party classifier (EfficientNet-B3, 1-channel input, Dropout(0.4)+Linear(1536,4)).
    Built with weights=None: the checkpoint supplies every weight, so no ImageNet download happens."""

    def __init__(self, num_classes: int = 4):
        super().__init__()
        from torchvision.models import efficientnet_b3
        backbone = efficientnet_b3(weights=None)
        old = backbone.features[0][0]
        backbone.features[0][0] = nn.Conv2d(1, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
        backbone.classifier = nn.Sequential(nn.Dropout(p=0.4), nn.Linear(backbone.classifier[1].in_features, num_classes))
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class _Service:
    def __init__(self):
        self.lock = threading.Lock()
        self.loaded = False
        self.error: str | None = None
        self.device = pick_device()
        self.model = None

    def ensure(self) -> bool:
        with self.lock:
            if not self.loaded and self.error is None:
                try:
                    self._load()
                    self.loaded = True
                except Exception as e:  # reported verbatim; never replaced by another model
                    self.error = f"{type(e).__name__}: {e}"
            return self.loaded


class GeneratorService(_Service):
    name = "cWGAN-GP generator"
    meta: dict = {}

    def _load(self):
        p = config.CWGAN_CKPT
        if not p.exists():
            raise FileNotFoundError(f"checkpoint not found: {self._rel(p)}")
        ck = torch.load(p, map_location="cpu", weights_only=True)
        g = CondGenerator()
        g.load_state_dict(ck["G"], strict=True)
        g.to(self.device).eval()
        cfg = ck.get("config", {})
        self.meta = {"checkpoint": self._rel(p), "generator_updates": int(ck.get("gen_updates", -1)),
                     "trained_on_images": cfg.get("images_used"), "image_size": cfg.get("image_size"),
                     "z_dim": cfg.get("z_dim"), "class_to_idx": cfg.get("class_to_idx"), "classifier_guidance": cfg.get("classifier_guidance"),
                     "train_partition": "cluster-disjoint training partition (de-duplicated experiment)"}
        if self.meta["class_to_idx"] != {c: i for i, c in enumerate(config.CLASSES)}:
            raise ValueError("checkpoint class mapping differs from the application's class order")
        self.model = g

    @staticmethod
    def _rel(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(config.PROJECT_ROOT))
        except ValueError:
            return p.name

    def generate(self, cls: str, n: int, seed: int | None):
        idx = config.CLASSES.index(cls)
        seed = int(seed) if seed is not None else int(np.random.SeedSequence().entropy % (2**31))
        gen = torch.Generator().manual_seed(seed)
        z = torch.randn(n, Z_DIM, generator=gen).to(self.device)   # latents drawn on CPU so a seed gives the same images on any device
        y = torch.full((n,), idx, dtype=torch.long, device=self.device)
        with self.lock, torch.no_grad():
            x = self.model(z, y).detach().cpu()
        arr = ((x.squeeze(1).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        out = []
        for a in arr:
            buf = io.BytesIO(); Image.fromarray(a, "L").save(buf, "PNG")
            out.append("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode())
        return seed, out


class ClassifierService(_Service):
    name = "Third-party EfficientNet-B3 brain-tumour classifier"
    mean, std = 0.1856, 0.1892

    def _load(self):
        p = config.CLASSIFIER_CKPT
        if not p.exists():
            raise FileNotFoundError("classifier weights not found (external_models/brain_tumor_classifier/weights/best_model.pth)")
        digest = sha256_file(p)
        if digest != config.CLASSIFIER_SHA256:
            raise ValueError(f"classifier weights SHA-256 {digest[:12]}... does not match the recorded {config.CLASSIFIER_SHA256[:12]}...")
        m = BrainTumorEfficientNet()
        m.load_state_dict(torch.load(p, map_location="cpu", weights_only=True), strict=True)
        m.to(self.device).eval()
        self.model = m

    def predict(self, img: Image.Image):
        g = img.convert("L").resize((224, 224), Image.BILINEAR)         # Grayscale(1) -> Resize((224,224)) as in the model's own pipeline
        t = torch.from_numpy(np.asarray(g, dtype=np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
        t = ((t - self.mean) / self.std).to(self.device)
        with self.lock, torch.no_grad():
            probs = F.softmax(self.model(t), dim=1).squeeze(0).cpu().numpy()
        return {c: float(p) for c, p in zip(config.CLASSES, probs)}


generator = GeneratorService()
classifier = ClassifierService()
