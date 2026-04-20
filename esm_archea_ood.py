"""
run_idp_esm2_on_bender.py
==========================================================
Evaluates IDP-ESM2-8M and IDP-ESM2-150M on BENDER test set
and OOD Archaea, replicating GeoGraph's evaluation protocol:

    1. Load frozen IDP-ESM2 backbone from HuggingFace
    2. Extract mean-pooled embeddings for idrome train sequences
    3. Train a shallow GeoHead MLP on idrome train embeddings
       (identical protocol to GeoGraph Table 1)
    4. Evaluate on:
         a) idrome test set  (in-distribution comparison)
         b) BENDER test set  (cross-kingdom generalization)
         c) OOD Archaea      (held-out kingdom)

GeoHead architecture (matches GeoGraph Appendix A):
    Linear(hidden_dim, 128) → SiLU → Dropout(0.1) → Linear(128, 4)
    4 targets: rg, ree, nu, delta (A0 excluded — not in idrome)

Usage
-----
    # install dependencies first:
    pip install transformers huggingface_hub torch pandas numpy tqdm

    python run_idp_esm2_on_bender.py \\
        --idrome    idrome_clustered.csv \\
        --bender    merged.csv \\
        --out       idp_esm2_results/ \\
        --models    8M 150M

Dependencies
------------
    transformers >= 4.30
    huggingface_hub
    torch, numpy, pandas, tqdm
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from collections import deque
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

TARGETS = ["rg", "ree", "nu", "delta"]

# idrome column mapping
IDROME_COL_MAP = {
    "fasta":   "sequence",
    "Rg/nm":   "rg",
    "Ree/nm":  "ree",
    "nu":      "nu",
    "Delta":   "delta",
}

# IDP-ESM2 model configs
ESM2_MODELS = {
    "8M":   {
        "hf_id":      "InstaDeepAI/IDP-ESM2-8M",
        "tokenizer":  "facebook/esm2_t6_8M_UR50D",
        "hidden_dim": 320,
    },
    "150M": {
        "hf_id":      "InstaDeepAI/IDP-ESM2-150M",
        "tokenizer":  "facebook/esm2_t12_35M_UR50D",  # 150M uses 640-dim
        "hidden_dim": 640,
    },
}

MAX_SEQ_LEN  = 256   # matching GeoGraph evaluation protocol
BATCH_SIZE   = 32
EMBED_BATCH  = 32    # batch size for embedding extraction
HEAD_EPOCHS  = 200
HEAD_PATIENCE = 20
HEAD_LR      = 3e-3  # GeoGraph uses 3e-3 for GeoHead training
HEAD_DROPOUT = 0.1


# ─────────────────────────────────────────────
# GeoHead MLP (matches GeoGraph architecture)
# ─────────────────────────────────────────────

class GeoHead(nn.Module):
    """
    Shallow MLP prediction head — identical to GeoGraph's FeaturesHead.
    Input: mean-pooled ESM2 embeddings (hidden_dim,)
    Output: 4 geometric targets (rg, ree, nu, delta)
    """
    def __init__(self, hidden_dim, n_out=4, dropout=HEAD_DROPOUT):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_out),
        )

    def forward(self, x):
        return self.head(x)

    def n_params(self):
        return sum(p.numel() for p in self.parameters()
                   if p.requires_grad)


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_idrome(path, max_seq_len=MAX_SEQ_LEN):
    """Load idrome CSV, rename columns, filter to ≤256 residues."""
    df = pd.read_csv(path)
    df = df.rename(columns=IDROME_COL_MAP)
    df = df.dropna(subset=["sequence"] + TARGETS)
    n_before = len(df)
    df = df[df["sequence"].str.len() <= max_seq_len].reset_index(drop=True)
    print(f"idrome: {n_before} → {len(df)} sequences (≤{max_seq_len} aa)")
    return df


def load_bender(path, max_seq_len=MAX_SEQ_LEN):
    """Load BENDER merged CSV."""
    df = pd.read_csv(path)
    df = df.dropna(subset=["sequence"] + TARGETS)
    n_before = len(df)
    df = df[df["sequence"].str.len() <= max_seq_len].reset_index(drop=True)
    print(f"BENDER: {n_before} → {len(df)} sequences (≤{max_seq_len} aa)")
    return df


def make_splits(df, train_frac=0.80, val_frac=0.10, seed=42):
    """Cluster-aware split if cluster_id present, else random."""
    if "cluster_id" in df.columns:
        rng = np.random.default_rng(seed)
        clusters = df["cluster_id"].unique()
        rng.shuffle(clusters)
        n  = len(clusters)
        n1 = int(train_frac * n)
        n2 = int(val_frac * n)
        tc = set(clusters[:n1])
        vc = set(clusters[n1:n1+n2])
        ec = set(clusters[n1+n2:])
        train = df[df["cluster_id"].isin(tc)].reset_index(drop=True)
        val   = df[df["cluster_id"].isin(vc)].reset_index(drop=True)
        test  = df[df["cluster_id"].isin(ec)].reset_index(drop=True)
        print(f"Cluster-aware split: "
              f"Train:{len(train)} Val:{len(val)} Test:{len(test)}")
    else:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(df))
        n   = len(idx)
        n1  = int(train_frac * n)
        n2  = int(val_frac * n)
        train = df.iloc[idx[:n1]].reset_index(drop=True)
        val   = df.iloc[idx[n1:n1+n2]].reset_index(drop=True)
        test  = df.iloc[idx[n1+n2:]].reset_index(drop=True)
        print(f"Random split: "
              f"Train:{len(train)} Val:{len(val)} Test:{len(test)}")
    return train, val, test


# ─────────────────────────────────────────────
# EMBEDDING EXTRACTION
# ─────────────────────────────────────────────

def load_esm2(model_key, device):
    """Load frozen IDP-ESM2 backbone and tokenizer."""
    cfg = ESM2_MODELS[model_key]
    print(f"\nLoading IDP-ESM2-{model_key} from {cfg['hf_id']} ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer"])
    model     = AutoModel.from_pretrained(cfg["hf_id"])
    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    print(f"  Hidden dim: {cfg['hidden_dim']}  "
          f"Params: {sum(p.numel() for p in model.parameters()):,}")
    return tokenizer, model, cfg["hidden_dim"]


@torch.no_grad()
def extract_embeddings(sequences, tokenizer, esm_model, device,
                       batch_size=EMBED_BATCH, max_len=MAX_SEQ_LEN):
    """
    Extract mean-pooled ESM2 embeddings for a list of sequences.
    Matches GeoGraph's mean-pooling over non-padding tokens.
    """
    all_embs = []
    for i in tqdm(range(0, len(sequences), batch_size),
                  desc="Extracting embeddings"):
        batch = sequences[i:i+batch_size]
        # truncate to max_len before tokenizing
        batch = [s[:max_len] for s in batch]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len + 2,  # +2 for special tokens
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = esm_model(**inputs)
        hidden  = outputs.last_hidden_state  # (B, L, D)

        # mean pool over non-padding tokens (excluding special tokens)
        # attention_mask: 1 for real tokens, 0 for padding
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        emb  = (hidden * mask).sum(1) / mask.sum(1)
        all_embs.append(emb.cpu())

    return torch.cat(all_embs, dim=0)  # (N, D)


# ─────────────────────────────────────────────
# EMBEDDING DATASET
# ─────────────────────────────────────────────

class EmbeddingDataset(Dataset):
    """Dataset of precomputed embeddings + targets."""
    def __init__(self, embeddings, targets_df, target_mean, target_std):
        self.emb    = embeddings
        self.tgt_raw = torch.tensor(
            targets_df[TARGETS].values.astype(float),
            dtype=torch.float32)
        self.mean   = torch.tensor(target_mean, dtype=torch.float32)
        self.std    = torch.tensor(target_std,  dtype=torch.float32)
        self.tgt    = (self.tgt_raw - self.mean) / self.std
        self.tgt    = torch.nan_to_num(self.tgt, nan=0.0)

    def __len__(self):
        return len(self.emb)

    def __getitem__(self, idx):
        return {"emb": self.emb[idx], "targets": self.tgt[idx],
                "targets_raw": self.tgt_raw[idx]}


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

def train_geohead(hidden_dim, train_emb, val_emb,
                  train_df, val_df, out_dir, name,
                  epochs=HEAD_EPOCHS, patience=HEAD_PATIENCE,
                  lr=HEAD_LR, device=None):
    """Train GeoHead MLP on frozen ESM2 embeddings."""
    device = device or torch.device("cpu")

    # normalisation stats from train set
    t_mean = train_df[TARGETS].mean().values
    t_std  = train_df[TARGETS].std().clip(lower=1e-6).values

    train_ds = EmbeddingDataset(train_emb, train_df, t_mean, t_std)
    val_ds   = EmbeddingDataset(val_emb,   val_df,   t_mean, t_std)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=2)

    model = GeoHead(hidden_dim).to(device)
    opt   = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=epochs)
    best  = float("inf")
    wait  = 0
    ckpt  = os.path.join(out_dir, f"{name}_geohead_best.pt")
    val_window = deque(maxlen=3)

    print(f"\nTraining GeoHead for {name} ({model.n_params():,} params)")
    print(f"{'Ep':>4} {'Trn':>8} {'Val':>8} {'nu_R2':>8}")
    print("─" * 36)

    for ep in range(epochs):
        model.train()
        tl = 0
        for b in train_dl:
            pred = model(b["emb"].to(device))
            loss = F.mse_loss(pred, b["targets"].to(device))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tl += loss.item()
        tl /= len(train_dl)

        model.eval()
        vl, vp, vt_raw = 0, [], []
        with torch.no_grad():
            for b in val_dl:
                pred = model(b["emb"].to(device))
                vl  += F.mse_loss(pred, b["targets"].to(device)).item()
                # denormalize for R² on raw scale
                p_raw = pred.cpu() * torch.tensor(t_std) + torch.tensor(t_mean)
                vp.append(p_raw)
                vt_raw.append(b["targets_raw"])
        vl /= len(val_dl)
        vp  = torch.cat(vp);  vt_raw = torch.cat(vt_raw)

        # nu R² on val
        nu_idx = TARGETS.index("nu")
        y, yh  = vt_raw[:, nu_idx].numpy(), vp[:, nu_idx].numpy()
        nu_r2  = 1 - ((y-yh)**2).sum() / (((y-y.mean())**2).sum() + 1e-10)

        sched.step()
        print(f"{ep+1:>4} {tl:>8.4f} {vl:>8.4f} {nu_r2:>8.3f}")

        val_window.append(vl)
        smoothed = sum(val_window) / len(val_window)
        if smoothed < best:
            best = smoothed; wait = 0
            torch.save({"state_dict": model.state_dict(),
                        "t_mean": t_mean, "t_std": t_std}, ckpt)
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stop @ ep {ep+1}"); break

    ckpt_data = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt_data["state_dict"])
    return model, ckpt_data["t_mean"], ckpt_data["t_std"]


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def r2_scores(preds, targets):
    """Coefficient of determination for each target."""
    out = {}
    for i, name in enumerate(TARGETS):
        y, yh  = targets[:, i].numpy(), preds[:, i].numpy()
        ss_res = ((y - yh)**2).sum()
        ss_tot = ((y - y.mean())**2).sum() + 1e-10
        out[name] = float(1 - ss_res / ss_tot)
    return out


@torch.no_grad()
def evaluate_on_df(model, embeddings, df, t_mean, t_std,
                   split_name, device):
    """Run GeoHead inference and compute R² scores."""
    model.eval()
    t_mean_t = torch.tensor(t_mean, dtype=torch.float32)
    t_std_t  = torch.tensor(t_std,  dtype=torch.float32)

    preds = []
    for i in range(0, len(embeddings), BATCH_SIZE):
        emb  = embeddings[i:i+BATCH_SIZE].to(device)
        pred = model(emb).cpu()
        preds.append(pred * t_std_t + t_mean_t)
    preds = torch.cat(preds)

    targets = torch.tensor(df[TARGETS].values.astype(float),
                           dtype=torch.float32)
    r2 = r2_scores(preds, targets)

    print(f"\n── {split_name} ({len(df)} sequences) ──────────────────")
    print(f"  {'Target':<12} {'R²':>6}")
    print(f"  {'──':<12} {'──':>6}")
    for tgt, score in r2.items():
        print(f"  {tgt:<12} {score:>6.4f}")

    return r2


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run(idrome_csv, bender_csv, out_dir, model_keys,
        seed=42, device_str=None):

    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed)

    if device_str:
        device = torch.device(device_str)
    elif torch.cuda.is_available():
        device = torch.device("cuda"); print("Device: CUDA")
    else:
        device = torch.device("cpu");  print("Device: CPU")

    # ── load data ────────────────────────────────────────────────
    print("\n=== Loading data ===")
    idrome_df = load_idrome(idrome_csv)
    bender_df = load_bender(bender_csv)

    idrome_train, idrome_val, idrome_test = make_splits(idrome_df, seed=seed)
    bender_ood  = bender_df[bender_df["kingdom"] == "Archaea"].copy()
    bender_test = bender_df[bender_df["kingdom"] != "Archaea"].copy()

    # filter bender test to same cluster split used by KESTREL
    # (use all non-Archaea sequences as test — consistent with cross-kingdom eval)
    print(f"\nBENDER test (non-Archaea): {len(bender_test)} sequences")
    print(f"BENDER OOD (Archaea):      {len(bender_ood)} sequences")

    all_results = {}

    for model_key in model_keys:
        print(f"\n{'='*60}")
        print(f"MODEL: IDP-ESM2-{model_key}")
        print(f"{'='*60}")

        name = f"IDP-ESM2-{model_key}"

        # ── load ESM2 backbone ───────────────────────────────────
        tokenizer, esm_model, hidden_dim = load_esm2(model_key, device)

        # ── extract embeddings ───────────────────────────────────
        print("\nExtracting idrome embeddings...")
        idrome_train_emb = extract_embeddings(
            idrome_train["sequence"].tolist(), tokenizer, esm_model, device)
        idrome_val_emb   = extract_embeddings(
            idrome_val["sequence"].tolist(),   tokenizer, esm_model, device)
        idrome_test_emb  = extract_embeddings(
            idrome_test["sequence"].tolist(),  tokenizer, esm_model, device)

        print("\nExtracting BENDER embeddings...")
        bender_test_emb = extract_embeddings(
            bender_test["sequence"].tolist(), tokenizer, esm_model, device)
        bender_ood_emb  = extract_embeddings(
            bender_ood["sequence"].tolist(),  tokenizer, esm_model, device)

        # free GPU memory from backbone
        esm_model.cpu()
        torch.cuda.empty_cache()

        # ── train GeoHead on idrome ──────────────────────────────
        model, t_mean, t_std = train_geohead(
            hidden_dim,
            idrome_train_emb, idrome_val_emb,
            idrome_train,     idrome_val,
            out_dir, name, device=device)

        # ── evaluate ─────────────────────────────────────────────
        results = {}
        results["idrome_test"] = evaluate_on_df(
            model, idrome_test_emb, idrome_test,
            t_mean, t_std, f"{name} — idrome TEST", device)

        results["bender_test"] = evaluate_on_df(
            model, bender_test_emb, bender_test,
            t_mean, t_std, f"{name} — BENDER TEST", device)

        results["ood_archaea"] = evaluate_on_df(
            model, bender_ood_emb, bender_ood,
            t_mean, t_std, f"{name} — OOD Archaea", device)

        all_results[name] = results

        # save per-model results
        rows = []
        for split, r2 in results.items():
            rows.append({"model": name, "split": split, **r2})
        pd.DataFrame(rows).to_csv(
            os.path.join(out_dir, f"{name}_results.csv"), index=False)

    # ── summary table ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY — nu R² across splits")
    print(f"{'Model':25s} {'idrome test':>12} {'BENDER test':>12} {'OOD Archaea':>12}")
    print("─" * 65)
    for name, results in all_results.items():
        print(f"{name:25s} "
              f"{results['idrome_test']['nu']:>12.4f} "
              f"{results['bender_test']['nu']:>12.4f} "
              f"{results['ood_archaea']['nu']:>12.4f}")

    # reference numbers
    print("─" * 65)
    print(f"{'GeoGraph (ref)':25s} {'0.8875':>12} {'—':>12} {'0.667':>12}")
    print(f"{'KESTREL-CrossKingdom':25s} {'—':>12} {'0.825':>12} {'0.768':>12}")

    # save full summary
    rows = []
    for name, results in all_results.items():
        for split, r2 in results.items():
            rows.append({"model": name, "split": split, **r2})
    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, "all_results.csv"), index=False)
    print(f"\nResults saved to {out_dir}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Evaluate IDP-ESM2 on BENDER via trained GeoHead")
    p.add_argument("--idrome",   required=True,
                   help="Path to idrome_clustered.csv")
    p.add_argument("--bender",   required=True,
                   help="Path to BENDER merged.csv")
    p.add_argument("--out",      default="idp_esm2_results/",
                   help="Output directory")
    p.add_argument("--models",   nargs="+", default=["8M", "150M"],
                   choices=["8M", "150M"],
                   help="Which IDP-ESM2 models to run (default: both)")
    p.add_argument("--seed",     default=42, type=int)
    p.add_argument("--device",   default=None,
                   help="Force device (cuda/cpu)")
    a = p.parse_args()
    run(a.idrome, a.bender, a.out, a.models, a.seed, a.device)

# ─────────────────────────────────────────────
# USAGE
# ─────────────────────────────────────────────
#
# Install:
#   pip install transformers huggingface_hub torch pandas numpy tqdm
#
# Run both models:
#   python run_idp_esm2_on_bender.py \\
#       --idrome idrome_clustered.csv \\
#       --bender merged.csv \\
#       --out    idp_esm2_results/
#
# Run 8M only (faster):
#   python run_idp_esm2_on_bender.py \\
#       --idrome idrome_clustered.csv \\
#       --bender merged.csv \\
#       --out    idp_esm2_results/ \\
#       --models 8M
#
# Output files:
#   idp_esm2_results/IDP-ESM2-8M_results.csv
#   idp_esm2_results/IDP-ESM2-150M_results.csv
#   idp_esm2_results/all_results.csv
#   idp_esm2_results/IDP-ESM2-8M_geohead_best.pt
#   idp_esm2_results/IDP-ESM2-150M_geohead_best.pt
# ─────────────────────────────────────────────
