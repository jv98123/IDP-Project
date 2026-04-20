"""
KESTREL-Human: KESTREL trained on idrome (human IDRs only)
===========================================================
Benchmark experiment for NeurIPS D&B submission.

Purpose
-------
Train KESTREL on the same data as GeoGraph (idrome, human IDRs only)
to isolate the architecture contribution from the data contribution:

    KESTREL-Human vs GeoGraph          → architecture comparison
    KESTREL-CrossKingdom vs KESTREL-Human → data diversity contribution

Targets (geometric only, 4)
----------------------------
    rg     - radius of gyration (nm)       ← idrome: "Rg/nm"
    ree    - end-to-end distance (nm)      ← idrome: "Ree/nm"
    nu     - Flory scaling exponent        ← idrome: "nu"
    delta  - asphericity                   ← idrome: "Delta"

A0 is excluded: idrome has no A0 column and GeoGraph's A0 is fit
directly from pairwise inter-residue distances (equation 5 in the
GeoGraph paper), not derivable as Rg / N^nu. The 4-target model is
directly comparable to GeoGraph on the same supervised quantities.

idrome column mapping
---------------------
    seq_name     → (identifier, not used)
    UniProt_ID   → (identifier, not used)
    nu           → nu
    Delta        → delta
    Rg/nm        → rg
    Ree/nm       → ree
    fasta        → sequence
    fK           → f_pos  (physchem)
    fR           → f_R
    fE           → f_E
    fD           → f_D
    faro         → f_aro
    mean_lambda  → lambda_avg
    shd          → SHD
    scd          → SCD
    kappa        → kappa
    fcr          → FCR
    ncpr         → NCPR
    QCDpred      → qcdpred

idrome does NOT have a kingdom column — all sequences are Human.
The kingdom embedding is disabled (zero vector for all samples).

No OOD split — idrome is human-only, no held-out kingdom.
Train/val/test = 80/10/10 random split (no cluster_id in idrome).
NOTE: GeoGraph uses a similarity-based split (MMseqs2, 70% seq ID).
A random split may slightly inflate scores — see make_splits_random()
for details. Sequences are filtered to ≤256 residues matching GeoGraph.

Usage
-----
    python kestrel_human.py --data idrome.csv --out kestrel_human_output/

    # compare on GeoGraph's test set (if you have the split indices):
    python kestrel_human.py --data idrome.csv --out kestrel_human_output/ \\
        --test_ids path/to/test_ids.txt

Dependencies
------------
    pip install torch numpy pandas tqdm
    (same as kestrel.py)
"""

import os
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import pandas as pd
from collections import deque

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

AA_VOCAB   = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX  = {aa: i for i, aa in enumerate(AA_VOCAB)}
VOCAB_SIZE = 20

# 4 directly supervised geometric targets — matches GeoGraph output.
# A0 excluded: not in idrome; GeoGraph's A0 is fit from inter-residue
# distances and is not equivalent to the derived Rg / N^nu.
GEO_TARGETS = ["rg", "ree", "nu", "delta"]
ALL_TARGETS = GEO_TARGETS          # 4 targets, no graph topology, no A0

# idrome → canonical column name mapping
IDROME_COL_MAP = {
    "fasta":       "sequence",
    "Rg/nm":       "rg",
    "Ree/nm":      "ree",
    "nu":          "nu",
    "Delta":       "delta",
    # physchem
    "mean_lambda": "lambda_avg",
    "shd":         "SHD",
    "scd":         "SCD",
    "kappa":       "kappa",
    "fcr":         "FCR",
    "ncpr":        "NCPR",
    "faro":        "f_aro",
    "QCDpred":     "qcdpred",
    "fK":          "f_pos",
    "fR":          "f_R",
    "fE":          "f_E",
    "fD":          "f_D",
}

# physchem columns available after mapping
# (subset of the full 13 used in cross-kingdom KESTREL)
PHYSCHEM_COLS = [
    "kappa", "SCD", "SHD", "FCR", "NCPR",
    "lambda_avg", "f_aro", "qcdpred",
    "f_pos", "f_R", "f_E", "f_D",
]


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_idrome(path, max_seq_len=256):
    """
    Load and normalise an idrome CSV into the canonical column names
    used by KESTREL.

    Sequences are filtered to max_seq_len=256 residues to match the
    evaluation protocol used by GeoGraph (Appendix A.3), ensuring a
    fair like-for-like comparison.

    Parameters
    ----------
    path        : str  path to idrome CSV
    max_seq_len : int  maximum sequence length — sequences longer than
                       this are excluded (default 256, matching GeoGraph)

    Returns
    -------
    pd.DataFrame with columns: sequence, rg, ree, nu, delta,
                                + available physchem columns
    """
    df = pd.read_csv(path)

    # rename to canonical names
    df = df.rename(columns=IDROME_COL_MAP)

    # add kingdom column as "Human" (kingdom embedding → zero vector always)
    df["kingdom"] = "Human"

    # drop rows missing any required target or sequence
    df = df.dropna(subset=["sequence", "rg", "ree", "nu", "delta"])

    # filter to max 256 residues — matches GeoGraph's evaluation protocol
    # (GeoGraph Appendix A.3: "filtered the dataset to sequences with a
    #  maximum length of 256 residues")
    n_before = len(df)
    df = df[df["sequence"].str.len() <= max_seq_len].reset_index(drop=True)
    n_after = len(df)
    print(f"Loaded {n_before} idrome sequences, "
          f"{n_after} after filtering to ≤{max_seq_len} residues "
          f"({n_before - n_after} excluded)")

    return df


def make_splits_random(df, train_frac=0.80, val_frac=0.10, seed=42):
    """
    Random 80/10/10 split — idrome has no cluster_id.
    No OOD split (all sequences are human).

    NOTE ON SPLIT METHODOLOGY
    -------------------------
    GeoGraph uses a sequence-similarity-based split via MMseqs2
    (min_seq_id=0.7, coverage=0.8, cov_mode=1) to ensure test sequences
    are genuinely novel relative to training. This random split does not
    enforce that constraint and may result in slightly optimistic test
    scores due to sequence leakage between splits.

    For a fully fair comparison with GeoGraph, replace this function
    with a cluster-based split using MMseqs2 cluster IDs. If cluster_id
    is available in the idrome CSV (e.g. from a prior MMseqs2 run), pass
    it to make_splits() from kestrel.py instead.

    In practice the effect is modest for rg/ree (which are largely
    length-determined) but may inflate nu and delta scores by ~0.01-0.03.
    This difference is noted explicitly in the paper.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n   = len(idx)
    n1  = int(train_frac * n)
    n2  = int(val_frac   * n)
    train_df = df.iloc[idx[:n1]].reset_index(drop=True)
    val_df   = df.iloc[idx[n1:n1+n2]].reset_index(drop=True)
    test_df  = df.iloc[idx[n1+n2:]].reset_index(drop=True)
    print(f"Train:{len(train_df)} Val:{len(val_df)} Test:{len(test_df)}")
    print("NOTE: random split — see make_splits_random() docstring "
          "for comparison with GeoGraph's similarity-based split.")
    return {"train": train_df, "val": val_df, "test": test_df}


def make_splits_cluster(df, train_frac=0.80, val_frac=0.10, seed=42):
    """
    Cluster-aware 80/10/10 split using cluster_id column.
    Mirrors make_splits() in kestrel.py. Requires cluster_id to be present
    — generated by prepare_idrome.py using MMseqs2 with GeoGraph parameters
    (min_seq_id=0.7, coverage=0.8, cov_mode=1).

    This split ensures test sequences are ≤70% identical to any training
    sequence, making the comparison with GeoGraph methodologically equivalent.
    """
    rng      = np.random.default_rng(seed)
    clusters = df["cluster_id"].unique()
    rng.shuffle(clusters)
    n  = len(clusters)
    n1 = int(train_frac * n)
    n2 = int(val_frac   * n)
    tc = set(clusters[:n1])
    vc = set(clusters[n1:n1+n2])
    ec = set(clusters[n1+n2:])
    train_df = df[df["cluster_id"].isin(tc)].reset_index(drop=True)
    val_df   = df[df["cluster_id"].isin(vc)].reset_index(drop=True)
    test_df  = df[df["cluster_id"].isin(ec)].reset_index(drop=True)
    print(f"Cluster-aware split: "
          f"Train:{len(train_df)} Val:{len(val_df)} Test:{len(test_df)} "
          f"({len(clusters)} clusters)")
    return {"train": train_df, "val": val_df, "test": test_df}




class IDPDatasetGeo(Dataset):
    """
    Geometric-targets-only dataset for idrome.
    Kingdom embedding is always zero (single organism).
    """
    def __init__(self, df, max_len=256, target_stats=None):
        self.df      = df.reset_index(drop=True)
        self.max_len = max_len

        if target_stats is None:
            self.tmean = df[ALL_TARGETS].mean()
            self.tstd  = df[ALL_TARGETS].std().clip(lower=1e-6)
        else:
            self.tmean = target_stats["mean"]
            self.tstd  = target_stats["std"]

    @property
    def target_stats(self):
        return {"mean": self.tmean, "std": self.tstd}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = str(row["sequence"])[:self.max_len]
        L   = len(seq)

        # one-hot encode
        oh   = torch.zeros(self.max_len, VOCAB_SIZE)
        mask = torch.zeros(self.max_len, dtype=torch.bool)
        for i, aa in enumerate(seq):
            j = AA_TO_IDX.get(aa, -1)
            if j >= 0:
                oh[i, j] = 1.0
        mask[:L] = True

        # normalized targets (5 geometric only)
        raw    = row[ALL_TARGETS].values.astype(float)
        normed = (raw - self.tmean.values) / self.tstd.values
        tgt    = torch.tensor(normed, dtype=torch.float32)
        tgt    = torch.nan_to_num(tgt, nan=0.0)

        # physchem for ablation
        if all(c in row.index for c in PHYSCHEM_COLS):
            pc = row[PHYSCHEM_COLS].values.astype(float)
        else:
            pc = np.zeros(len(PHYSCHEM_COLS))
        physchem = torch.nan_to_num(
            torch.tensor(pc, dtype=torch.float32), nan=0.0)

        return {
            "sequence":    oh,            # (max_len, 20)
            "mask":        mask,           # (max_len,)
            "targets":     tgt,            # (5,)
            "physchem":    physchem,       # (12,)
            "kingdom":     "Human",
            # kingdom_idx = -1 always → zero embedding in KESTREL
            "kingdom_idx": torch.tensor(-1, dtype=torch.long),
            "length":      L,
        }


# ─────────────────────────────────────────────
# MODEL  (identical architecture to kestrel.py)
# ─────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256, dropout=0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(1)])


class SequenceEncoder(nn.Module):
    def __init__(self, d_model=256, n_heads=8,
                 n_layers=4, ffn_dim=512,
                 dropout=0.1, max_len=256):
        super().__init__()
        self.proj = nn.Linear(VOCAB_SIZE, d_model)
        self.pe   = PositionalEncoding(d_model, max_len, dropout)
        layer     = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=ffn_dim, dropout=dropout,
            batch_first=True, norm_first=True,
            activation="gelu",
        )
        self.enc  = nn.TransformerEncoder(
            layer, num_layers=n_layers,
            enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        x     = self.pe(self.proj(x))
        x     = self.enc(x, src_key_padding_mask=~mask)
        x     = self.norm(x)
        valid = mask.unsqueeze(-1).float()
        return (x * valid).sum(1) / valid.sum(1)


class PredHead(nn.Module):
    def __init__(self, in_dim, hidden=128, n_out=5, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_out),
        )
    def forward(self, x):
        return self.net(x)


KINGDOM_DIM = 16

class KestrelHuman(nn.Module):
    """
    KESTREL architecture, geometric targets only, no kingdom embedding
    (all sequences are human — kingdom is uninformative).

    Output: (B, 4)  [rg, ree, nu, delta]
    """
    def __init__(self, d_model=256, n_heads=8,
                 n_layers=4, ffn_dim=512,
                 dropout=0.1, max_len=256
                 ):
        super().__init__()
        self.encoder  = SequenceEncoder(
            d_model, n_heads, n_layers, ffn_dim, dropout, max_len)
        self.geo_head = PredHead(d_model, 128, len(GEO_TARGETS), dropout)
    def forward(self, seq, mask, kingdom_idx=None):
        # kingdom_idx accepted but ignored — API compatible with kestrel.py
        emb = self.encoder(seq, mask)
        return self.geo_head(emb)

    def n_params(self):
        return sum(p.numel() for p in self.parameters()
                   if p.requires_grad)

    @torch.no_grad()
    def predict(self, sequences, target_stats, device,
                max_len=256, batch_size=128):
        self.eval()
        preds = []
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]
            B = len(batch)
            oh   = torch.zeros(B, max_len, VOCAB_SIZE)
            mask = torch.zeros(B, max_len, dtype=torch.bool)
            for b, seq in enumerate(batch):
                seq = seq[:max_len]
                for j, aa in enumerate(seq):
                    k = AA_TO_IDX.get(aa, -1)
                    if k >= 0:
                        oh[b, j, k] = 1.0
                mask[b, :len(seq)] = True
            p = self(oh.to(device), mask.to(device)).cpu()
            mean = torch.tensor(target_stats["mean"].values,
                                dtype=torch.float32)
            std  = torch.tensor(target_stats["std"].values,
                                dtype=torch.float32)
            preds.append(p * std + mean)
        return pd.DataFrame(
            torch.cat(preds).numpy(), columns=ALL_TARGETS)


class PhyschemMLPGeo(nn.Module):
    """
    Ablation: physchem scalar features → 5 geometric targets.
    Uses the idrome physchem columns (12 features).
    """
    def __init__(self, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(len(PHYSCHEM_COLS), 128), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, len(ALL_TARGETS)),
        )
    def forward(self, physchem):
        return self.net(physchem)
    def n_params(self):
        return sum(p.numel() for p in self.parameters()
                   if p.requires_grad)


# ─────────────────────────────────────────────
# TRAINING + EVALUATION
# ─────────────────────────────────────────────

def get_device():
    if torch.backends.mps.is_available():
        print("Device: Mac MPS"); return torch.device("mps")
    if torch.cuda.is_available():
        print("Device: CUDA");    return torch.device("cuda")
    print("Device: CPU");         return torch.device("cpu")


def r2_scores(preds, targets):
    out = {}
    for i, name in enumerate(ALL_TARGETS):
        y, yh  = targets[:,i].numpy(), preds[:,i].numpy()
        ss_res = ((y - yh)**2).sum()
        ss_tot = ((y - y.mean())**2).sum() + 1e-10
        out[name] = float(1 - ss_res/ss_tot)
    return out


def step(model, batch, device, is_kestrel):
    if is_kestrel:
        preds = model(batch["sequence"].to(device),
                      batch["mask"].to(device),
                      batch["kingdom_idx"].to(device))
    else:
        preds = model(batch["physchem"].to(device))
    targets = batch["targets"].to(device)
    return F.mse_loss(preds, targets), preds.cpu(), targets.cpu()


def train_model(model, train_loader, val_loader,
                out_dir, name, epochs=100, patience=15,
                lr=5e-4, wd=1e-4, device=None,
                is_kestrel=True, smooth_window=3):
    device = device or get_device()
    model  = model.to(device)
    opt    = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched  = CosineAnnealingLR(opt, T_max=epochs)
    best   = float("inf")
    wait   = 0
    ckpt   = os.path.join(out_dir, f"{name}_best.pt")
    hist   = []
    val_window = deque(maxlen=smooth_window)

    print(f"\nTraining {name} ({model.n_params():,} params)")
    print(f"{'Ep':>4} {'Trn':>8} {'Val':>8} "
          f"{'Rg':>7} {'nu':>7} {'Δ':>7}")
    print("─" * 46)

    for ep in range(epochs):
        model.train()
        tl = 0
        for b in train_loader:
            loss, _, _ = step(model, b, device, is_kestrel)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tl += loss.item()
        tl /= len(train_loader)

        model.eval()
        vl, vp, vt, _ = evaluate(model, val_loader, device, is_kestrel)
        sched.step()
        r2 = r2_scores(vp, vt)
        hist.append({"epoch": ep+1, "train": tl, "val": vl, **r2})

        print(f"{ep+1:>4} {tl:>8.4f} {vl:>8.4f} "
              f"{r2['rg']:>7.3f} {r2['nu']:>7.3f} "
              f"{r2['delta']:>7.3f}")

        val_window.append(vl)
        smoothed_vl = sum(val_window) / len(val_window)

        if smoothed_vl < best:
            best = smoothed_vl; wait = 0
            torch.save(model.state_dict(), ckpt)
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stop @ ep {ep+1}"); break

    pd.DataFrame(hist).to_csv(
        os.path.join(out_dir, f"{name}_history.csv"), index=False)
    model.load_state_dict(
        torch.load(ckpt, map_location=device))
    return model


@torch.no_grad()
def evaluate(model, loader, device, is_kestrel=True):
    model.eval()
    ap, at, ak = [], [], []
    for b in loader:
        _, p, t = step(model, b, device, is_kestrel)
        ap.append(p); at.append(t)
        ak.extend(b["kingdom"])
    return (F.mse_loss(torch.cat(ap), torch.cat(at)).item(),
            torch.cat(ap), torch.cat(at), ak)


def full_eval(model, test_loader, out_dir, name, device,
              is_kestrel=True):
    print(f"\n{'='*55}\nEvaluation: {name}\n{'='*55}")
    _, p, t, _ = evaluate(model, test_loader, device, is_kestrel)
    r2 = r2_scores(p, t)

    print(f"\n── TEST ──────────────────────────────")
    for tgt, score in r2.items():
        print(f"  {tgt:30s}: {score:.4f}")

    pd.DataFrame([{"model": name, "split": "test", **r2}]).to_csv(
        os.path.join(out_dir, f"{name}_test_r2.csv"), index=False)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run(data_csv, out_dir="kestrel_human_output/", max_len=256,
        batch_size=256, epochs=100, patience=15,
        lr=5e-4, seed=42):

    torch.manual_seed(seed); np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    device = get_device()

    # load + normalise idrome, filtered to ≤256 residues (matching GeoGraph)
    df = load_idrome(data_csv, max_seq_len=max_len)

    # auto-detect split strategy:
    #   cluster_id present → similarity-based split (matches GeoGraph exactly)
    #   cluster_id absent  → random split (see make_splits_random() docstring)
    if "cluster_id" in df.columns:
        print("cluster_id detected — using similarity-based split "
              "(matches GeoGraph MMseqs2 protocol)")
        splits = make_splits_cluster(df, seed=seed)
    else:
        print("No cluster_id found — using random split. "
              "Run prepare_idrome.py first for a fair GeoGraph comparison.")
        splits = make_splits_random(df, seed=seed)

    stats = {
        "mean": splits["train"][ALL_TARGETS].mean(),
        "std":  splits["train"][ALL_TARGETS].std().clip(lower=1e-6),
    }

    # datasets — max_len=256 matching GeoGraph's evaluation protocol
    kw = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    def make_loader(split_df, shuffle):
        ds = IDPDatasetGeo(split_df, max_len, stats)
        return DataLoader(ds, shuffle=shuffle, **kw)

    train_dl = make_loader(splits["train"], True)
    val_dl   = make_loader(splits["val"],   False)
    test_dl  = make_loader(splits["test"],  False)

    # ── KESTREL-Human ────────────────────────────────────────────
    model = train_model(
        KestrelHuman(max_len=max_len),
        train_dl, val_dl, out_dir,
        "kestrel_human", epochs, patience, lr,
        device=device, is_kestrel=True, smooth_window=3)
    full_eval(model, test_dl, out_dir, "kestrel_human", device)

    # save stats for inference
    stats["mean"].to_csv(os.path.join(out_dir, "target_mean.csv"))
    stats["std"].to_csv( os.path.join(out_dir, "target_std.csv"))

    # ── PhyschemMLP ablation ──────────────────────────────────────
    pc_model = train_model(
        PhyschemMLPGeo(), train_dl, val_dl, out_dir,
        "physchem_mlp", epochs, patience=20, lr=lr,
        device=device, is_kestrel=False, smooth_window=3)
    full_eval(pc_model, test_dl, out_dir, "physchem_mlp", device,
              is_kestrel=False)

    # ── summary ──────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("SUMMARY TABLE (Test set) — 4 geometric targets")
    print(f"{'Model':20s} {'Rg':>7} {'Ree':>7} {'nu':>7} {'Δ':>7}")
    for name, fname in [("KESTREL-Human",  "kestrel_human_test_r2.csv"),
                         ("PhyschemMLP",    "physchem_mlp_test_r2.csv")]:
        fp = os.path.join(out_dir, fname)
        if os.path.exists(fp):
            r = pd.read_csv(fp).iloc[0]
            print(f"{name:20s} {r['rg']:>7.3f} {r['ree']:>7.3f} "
                  f"{r['nu']:>7.3f} {r['delta']:>7.3f}")

    print(f"\nGeoGraph (reference)   "
          f"{'0.894':>7} {'0.888':>7} {'0.756':>7} {'0.796':>7}")

    return model


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data",       required=True,
                   help="Path to idrome CSV")
    p.add_argument("--out",        default="kestrel_human_output/")
    p.add_argument("--max_len",    default=256,  type=int,
                   help="Max sequence length — must match GeoGraph (256) for fair comparison")
    p.add_argument("--batch_size", default=256,  type=int)
    p.add_argument("--epochs",     default=100,  type=int)
    p.add_argument("--patience",   default=15,   type=int)
    p.add_argument("--lr",         default=5e-4, type=float)
    p.add_argument("--seed",       default=42,   type=int)
    a = p.parse_args()
    run(a.data, a.out, a.max_len, a.batch_size,
        a.epochs, a.patience, a.lr, a.seed)

# ─────────────────────────────────────────────
# QUICK START
# ─────────────────────────────────────────────
#
# Train:
#   python kestrel_human.py --data idrome.csv --out kestrel_human_output/
#
# Inference:
#   from kestrel_human import KestrelHuman
#   import torch, pandas as pd
#
#   model = KestrelHuman()
#   model.load_state_dict(
#       torch.load("kestrel_human_output/kestrel_human_best.pt"))
#
#   stats = {
#       "mean": pd.read_csv("kestrel_human_output/target_mean.csv",
#                           index_col=0).squeeze(),
#       "std":  pd.read_csv("kestrel_human_output/target_std.csv",
#                           index_col=0).squeeze(),
#   }
#   seqs  = ["MAEADFKMVSEPVAHGVAEE", "MGKGTDMARAKARRLKGMK"]
#   preds = model.predict(seqs, stats, torch.device("cpu"))
#   print(preds)   # columns: rg, ree, nu, delta
#
# ─────────────────────────────────────────────
