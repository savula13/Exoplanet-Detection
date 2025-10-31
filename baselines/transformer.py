# transformer_exo.py
# PyTorch implementation inspired by: "Exoplanet Transit Candidate Identification in TESS Full-Frame Images via a Transformer-Based Algorithm"
# Uses the paper's preprocess, augmentation, conv-embedding and transformer encoder design.
# Key references from the paper: input length = 1000, conv-embedding with two 1D convs (first kernel=1),
# default encoder L=4, heads=4, dropout=0.1, GELU activation, avg pooling then MLP head.
# (See paper for full details.) :contentReference[oaicite:4]{index=4} :contentReference[oaicite:5]{index=5} :contentReference[oaicite:6]{index=6}

import math
import copy
import numpy as np
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random

# --------------------------
# CONFIGURABLE HYPERPARAMS
# --------------------------
# Model / embedding params (change these to scale the model)
SEQ_LEN = 1000               # standard sequence length per paper. :contentReference[oaicite:7]{index=7}
INPUT_FEATURES = 3           # flux, centroid magnitude, background (paper uses these three). :contentReference[oaicite:8]{index=8}

EMBED_DIM = 128              # d_emb in paper default 512 (table mentions demb options). :contentReference[oaicite:9]{index=9}
CONV1_OUT = 256              # intermediate conv channels (first conv maps m -> conv1_out)
CONV1_KERNEL = 1             # paper: first conv kernel k=1. :contentReference[oaicite:10]{index=10}
CONV2_OUT = EMBED_DIM        # second conv produces EMBED_DIM tokens (paper: second conv filters = d)
CONV2_KERNEL = 3             # you can change (paper doesn't specify for second explicitly, but 1D conv stack used)
PATCH_STRIDE = 1             # stride for convs (controls token count T)

NUM_ENCODER_LAYERS = 2       # l = 4 default in paper. :contentReference[oaicite:11]{index=11}
NUM_HEADS = 4                # heads = 4 default. :contentReference[oaicite:12]{index=12}
FFN_DIM = 512               # feedforward dim in transformer (often 4*embed_dim)
DROPOUT = 0.1                # dropout used per layer. :contentReference[oaicite:13]{index=13}

MLP_HIDDEN = 256             # hidden size for final MLP head
OUTPUT_DIM = 1               # binary classification logits

# Training params (make variables)
BATCH_SIZE = 8             # paper uses 120. :contentReference[oaicite:14]{index=14}
NUM_EPOCHS = 200             # per paper. :contentReference[oaicite:15]{index=15}
LR = 1e-3                    # paper initial LR. :contentReference[oaicite:16]{index=16}
LR_REDUCE_FACTOR = 0.8       # reduce by 20% when plateau (paper: reduce by 20%). :contentReference[oaicite:17]{index=17}
LR_PATIENCE = 7              # patience in epochs (paper: 5-10; choose 7)
WEIGHT_DECAY = 1e-5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# UTILITIES: Preprocessing
# --------------------------
def standardize_length(arr: np.ndarray, target_len: int = SEQ_LEN) -> np.ndarray:
    """
    Truncate to target_len or repeat the initial segment until reaching target_len.
    Matches the paper's described behaviour for sequences shorter than 1000.
    """
    if len(arr) >= target_len:
        return arr[:target_len].copy()
    # repeat the start until filled
    out = np.zeros(target_len, dtype=arr.dtype)
    i = 0
    while i < target_len:
        take = min(len(arr), target_len - i)
        out[i:i+take] = arr[:take]
        i += take
    return out

def normalize_to_minus1_1(x: np.ndarray) -> np.ndarray:
    """
    subtract mean, divide by std, then linearly map to [-1, 1]
    per the paper's description.
    """
    x = x.astype(np.float32)
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma == 0:
        sigma = 1.0
    z = (x - mu) / sigma
    # scale to [-1,1] by dividing by max abs value then clip
    maxabs = max(1e-6, np.max(np.abs(z)))
    z = z / maxabs
    z = np.clip(z, -1.0, 1.0)
    return z

def centroid_magnitude(xrow: np.ndarray, ycol: np.ndarray) -> np.ndarray:
    """r_centroid = sqrt(xrow^2 + ycol^2) as paper describes."""
    return np.sqrt(np.square(xrow) + np.square(ycol))

# --------------------------
# AUGMENTATIONS (paper: applied only to training)
# white noise (std sampled between 0 and mean of flux),
# random roll, split-and-swap, mirror
# --------------------------
def aug_add_white_noise(flux: np.ndarray):
    mean_flux = np.mean(np.abs(flux))
    sigma = random.uniform(0.0, float(mean_flux))
    noise = np.random.normal(0.0, sigma, size=flux.shape).astype(np.float32)
    return flux + noise

def aug_random_roll(arrs: List[np.ndarray]):
    shift = random.randrange(len(arrs[0]))
    return [np.roll(a, shift) for a in arrs]

def aug_split_and_swap(arrs: List[np.ndarray]):
    n = len(arrs[0])
    idx = random.randrange(1, n)  # split index
    out = []
    for a in arrs:
        a1 = a[:idx].copy()
        a2 = a[idx:].copy()
        out.append(np.concatenate([a2, a1]))
    return out

def aug_mirror(arrs: List[np.ndarray]):
    return [a[::-1].copy() for a in arrs]

AUG_FUNCTIONS = [
    ("white_noise", aug_add_white_noise),
    ("roll", aug_random_roll),
    ("split_swap", aug_split_and_swap),
    ("mirror", aug_mirror),
]

def apply_random_augmentation(flux, centroid, background):
    choice = random.choice(AUG_FUNCTIONS)
    name = choice[0]
    fn = choice[1]
    if name == "white_noise":
        flux = fn(flux)
        # leave centroid/background unchanged for this augmentation
        return flux, centroid, background
    else:
        arrs = fn([flux, centroid, background])
        return arrs[0], arrs[1], arrs[2]

# --------------------------
# DATASET wrapper example
# --------------------------
class LightCurveDataset(Dataset):
    """
    Expects a list/array of raw inputs per sample:
    Each sample: dict with keys: 'flux', 'centroid_row', 'centroid_col', 'background', 'label'
    Preprocessing done here to produce a numeric tensor of shape (SEQ_LEN, INPUT_FEATURES)
    """
    def __init__(self, samples: List[dict], augment: bool = False):
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        flux = np.asarray(s['flux'], dtype=np.float32)
        crow = np.asarray(s['centroid_row'], dtype=np.float32)
        ccol = np.asarray(s['centroid_col'], dtype=np.float32)
        bg = np.asarray(s['background'], dtype=np.float32)
        label = float(s['label'])

        # Standardize length
        flux = standardize_length(flux, SEQ_LEN)
        crow = standardize_length(crow, SEQ_LEN)
        ccol = standardize_length(ccol, SEQ_LEN)
        bg = standardize_length(bg, SEQ_LEN)

        # centroid magnitude
        rcent = centroid_magnitude(crow, ccol)

        # augment only training
        if self.augment:
            flux, rcent, bg = apply_random_augmentation(flux, rcent, bg)

        # normalize to -1..1 per variable
        flux = normalize_to_minus1_1(flux)
        rcent = normalize_to_minus1_1(rcent)
        bg = normalize_to_minus1_1(bg)

        # concatenate into shape (SEQ_LEN, INPUT_FEATURES)
        x = np.stack([flux, rcent, bg], axis=1).astype(np.float32)

        # convert to tensor (seq, feat) -> we will transpose to (feat, seq) for conv1d
        x = torch.from_numpy(x)  # shape (SEQ_LEN, 3)
        y = torch.tensor(label, dtype=torch.float32)

        return x, y

# --------------------------
# MODEL PARTS
# --------------------------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = SEQ_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # shape (max_len, d_model)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:seq_len].unsqueeze(0)

class ConvEmbedding(nn.Module):
    """
    Two-layer 1D conv embedding:
    - input shape expected: (batch, seq_len, INPUT_FEATURES)
    - we transpose to (batch, features, seq_len) for conv1d
    - first conv: kernel size = CONV1_KERNEL, maps INPUT_FEATURES -> CONV1_OUT
    - second conv: kernel size = CONV2_KERNEL, maps CONV1_OUT -> CONV2_OUT (EMBED_DIM)
    Output: (batch, seq_len', EMBED_DIM)
    """
    def __init__(self,
                 in_features: int = INPUT_FEATURES,
                 conv1_out: int = CONV1_OUT,
                 conv1_kernel: int = CONV1_KERNEL,
                 conv2_out: int = CONV2_OUT,
                 conv2_kernel: int = CONV2_KERNEL,
                 stride: int = PATCH_STRIDE):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, conv1_out, kernel_size=conv1_kernel, stride=stride, padding=conv1_kernel//2)
        self.bn1 = nn.BatchNorm1d(conv1_out)
        self.conv2 = nn.Conv1d(conv1_out, conv2_out, kernel_size=conv2_kernel, stride=1, padding=conv2_kernel//2)
        self.bn2 = nn.BatchNorm1d(conv2_out)
        self.act = nn.GELU()

    def forward(self, x):
        # x: (batch, seq_len, in_features)
        x = x.transpose(1, 2)  # (batch, in_features, seq_len)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)  # (batch, EMBED_DIM, seq_len)
        x = x.transpose(1, 2)  # (batch, seq_len, EMBED_DIM)
        return x

class ExoTransformer(nn.Module):
    def __init__(self,
                 seq_len: int = SEQ_LEN,
                 in_features: int = INPUT_FEATURES,
                 embed_dim: int = EMBED_DIM,
                 conv1_out: int = CONV1_OUT,
                 conv1_kernel: int = CONV1_KERNEL,
                 conv2_out: int = CONV2_OUT,
                 conv2_kernel: int = CONV2_KERNEL,
                 num_layers: int = NUM_ENCODER_LAYERS,
                 nhead: int = NUM_HEADS,
                 dim_feedforward: int = FFN_DIM,
                 dropout: float = DROPOUT,
                 mlp_hidden: int = MLP_HIDDEN,
                 output_dim: int = OUTPUT_DIM):
        super().__init__()
        self.embed = ConvEmbedding(in_features, conv1_out, conv1_kernel, conv2_out, conv2_kernel)
        self.posenc = SinusoidalPositionalEncoding(embed_dim, max_len=seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim,
                                                   nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   activation='gelu',
                                                   batch_first=True)  # batch_first: (B, S, D)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layernorm = nn.LayerNorm(embed_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)  # we will use pooling across seq dimension

        # MLP head
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, output_dim)
        )

    def forward(self, x):
        # x: (batch, seq_len, in_features)
        x = self.embed(x)                      # (batch, seq_len, embed_dim)
        x = self.posenc(x)                     # add positional encoding
        x = self.transformer_encoder(x)        # (batch, seq_len, embed_dim)
        x = self.layernorm(x)
        # average pooling across seq_len: transform to (batch, embed_dim, seq_len) for AdaptiveAvgPool1d
        x_t = x.transpose(1, 2)                # (batch, embed_dim, seq_len)
        pooled = self.pool(x_t).squeeze(-1)    # (batch, embed_dim)
        # final logits
        logits = self.mlp(pooled).squeeze(-1)  # (batch,)  single logit per sample
        return logits

# --------------------------
# TRAIN / EVALUATION helpers
# --------------------------
def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer, loss_fn, device=DEVICE):
    model.train()
    running_loss = 0.0
    for xb, yb in dataloader:
        xb = xb.to(device)        # shape (batch, seq_len, features)
        yb = yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
    return running_loss / len(dataloader.dataset)

@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, device=DEVICE):
    model.eval()
    preds = []
    targets = []
    for xb, yb in dataloader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        probs = torch.sigmoid(logits)
        preds.append(probs.cpu().numpy())
        targets.append(yb.cpu().numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    # compute basic metrics (AUC/F1 require scikit-learn; here return accuracy & placeholder)
    acc = ((preds >= 0.5) == targets).mean()
    return {"accuracy": float(acc), "preds": preds, "targets": targets}

# --------------------------
# TOP-LEVEL TRAINING FUNCTION (example)
# --------------------------
def train_model(train_samples, val_samples,
                model_params: dict = None,
                train_params: dict = None,
                device: torch.device = DEVICE):
    # model_params/train_params override defaults if provided
    if model_params is None:
        model_params = {}
    if train_params is None:
        train_params = {}

    model = ExoTransformer(**model_params).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=train_params.get("lr", LR), weight_decay=train_params.get("wd", WEIGHT_DECAY))
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=LR_REDUCE_FACTOR,
                                  patience=LR_PATIENCE)

    train_ds = LightCurveDataset(train_samples, augment=True)
    val_ds = LightCurveDataset(val_samples, augment=False)
    train_loader = DataLoader(train_ds, batch_size=train_params.get("batch_size", BATCH_SIZE), shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=train_params.get("batch_size", BATCH_SIZE), shuffle=False, num_workers=2, pin_memory=True)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(train_params.get("epochs", NUM_EPOCHS)):
        loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_res = evaluate(model, val_loader, device)
        val_acc = val_res["accuracy"]
        # Use validation accuracy to drive scheduler (paper reduces LR by 20% if no improvement after 5-10 epochs).
        scheduler.step(val_acc)
        print(f"Epoch {epoch+1}/{train_params.get('epochs', NUM_EPOCHS)} - Train loss: {loss:.4f} - Val acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    # load best
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {"best_val_acc": best_val_acc}

# --------------------------
# EXAMPLE USAGE (pseudo)
# --------------------------
if __name__ == "__main__":
    # Example: create toy samples (replace with real SPOC-derived arrays)
    # Each sample should include: 'flux', 'centroid_row', 'centroid_col', 'background', 'label'
    n_train = 200
    n_val = 40
    def make_random_sample(label):
        length = random.randint(700, 1300)
        flux = np.random.normal(1.0, 0.01, size=length).astype(np.float32)
        # insert a tiny box-transit if label==1 (toy)
        if label == 1:
            start = random.randint(50, length-50)
            depth = random.uniform(0.001, 0.02)
            dur = random.randint(1, 8)
            flux[start:start+dur] -= depth
        crow = np.random.normal(0.0, 0.01, size=length).astype(np.float32)
        ccol = np.random.normal(0.0, 0.01, size=length).astype(np.float32)
        bg = np.random.normal(0.0, 0.001, size=length).astype(np.float32)
        return {'flux': flux, 'centroid_row': crow, 'centroid_col': ccol, 'background': bg, 'label': label}

    train_samples = [make_random_sample(random.choice([0,1])) for _ in range(n_train)]
    val_samples = [make_random_sample(random.choice([0,1])) for _ in range(n_val)]

    model_params = {
        "seq_len": SEQ_LEN,
        "in_features": INPUT_FEATURES,
        "embed_dim": EMBED_DIM,
        "conv1_out": CONV1_OUT,
        "conv1_kernel": CONV1_KERNEL,
        "conv2_out": CONV2_OUT,
        "conv2_kernel": CONV2_KERNEL,
        "num_layers": NUM_ENCODER_LAYERS,
        "nhead": NUM_HEADS,
        "dim_feedforward": FFN_DIM,
        "dropout": DROPOUT,
        "mlp_hidden": MLP_HIDDEN,
    }
    train_params = {"lr": LR, "batch_size": BATCH_SIZE, "epochs": 10}  # for quick example use fewer epochs
    model, info = train_model(train_samples, val_samples, model_params=model_params, train_params=train_params)
    print("Trained. Best val acc:", info["best_val_acc"])
