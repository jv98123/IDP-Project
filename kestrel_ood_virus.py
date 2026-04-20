"""
KESTREL: Kingdom-aware Ensemble Sequence-To-pRopErty Learner

Dependencies:
    pip install torch numpy pandas tqdm
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

GEO_TARGETS = ["rg", "ree", "nu", "delta", "A0"]

GRAPH_TARGETS = [
    "global_efficiency",
    "fragmentation_index",
    "avg_clustering",
    "transitivity",
    "degree_assortativity",
]

ALL_TARGETS = GEO_TARGETS + GRAPH_TARGETS       # 10 targets

PHYSCHEM_COLS = [
    "kappa", "SCD", "SHD", "FCR", "NCPR",
    "lambda_avg", "f_aro", "f_hydrophobic",
    "f_pos", "f_neg", "hydropathy_avg",
    "pi_score", "qcdpred",
]

OOD_KINGDOM  = "Viruses"    # held out entirely — always uses zero embedding
KINGDOM_DIM  = 16           # embedding dimension


# ─────────────────────────────────────────────
# KINGDOM VOCABULARY
# ─────────────────────────────────────────────

def build_kingdom_vocab(train_df, ood_kingdom=OOD_KINGDOM):
    """
    Build an ordered kingdom→index mapping from the training split.
    The OOD kingdom is intentionally excluded so it always maps to
    the zero-vector fallback at inference.

    Returns
    -------
    dict  kingdom_name → int index (0-based)
    """
    kingdoms = sorted(
        k for k in train_df["kingdom"].unique()
        if k != ood_kingdom
    )
    return {k: i for i, k in enumerate(kingdoms)}


def save_kingdom_vocab(vocab, path):
    with open(path, "w") as f:
        for k, i in sorted(vocab.items(), key=lambda x: x[1]):
            f.write(f"{i}\t{k}\n")


def load_kingdom_vocab(path):
    vocab = {}
    with open(path) as f:
        for line in f:
            i, k = line.strip().split("\t", 1)
            vocab[k] = int(i)
    return vocab


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────

class IDPDataset(Dataset):
    """
    Parameters
    ----------
    df            : DataFrame for this split
    max_len       : sequences truncated/padded to this length
    target_stats  : {"mean": Series, "std": Series}
                    always computed from train split, passed to val/test/ood
    kingdom_vocab : dict mapping kingdom name → int index.
                    Unknown kingdoms (incl. OOD) return index = -1,
                    which the model maps to a zero embedding.
    """
    def __init__(self, df, max_len=256, target_stats=None,
                 kingdom_vocab=None):
        self.df           = df.reset_index(drop=True)
        self.max_len      = max_len
        self.kingdom_vocab = kingdom_vocab or {}

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

        # normalized targets
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

        # kingdom index: -1 for unknown/OOD → zero embedding in model
        kingdom_str = str(row.get("kingdom", "Unknown"))
        kingdom_idx = self.kingdom_vocab.get(kingdom_str, -1)

        return {
            "sequence":    oh,            # (max_len, 20)
            "mask":        mask,           # (max_len,)
            "targets":     tgt,            # (10,)
            "physchem":    physchem,       # (13,)
            "kingdom":     kingdom_str,
            "kingdom_idx": torch.tensor(kingdom_idx, dtype=torch.long),
            "length":      L,
        }


def make_splits(df, ood_kingdom=OOD_KINGDOM,
                train_frac=0.80, val_frac=0.10, seed=42):
    """
    1. Separate OOD kingdom entirely
    2. Cluster-aware 80/10/10 split on remainder
    """
    rng     = np.random.default_rng(seed)
    ood_df  = df[df["kingdom"] == ood_kingdom].copy()
    main_df = df[df["kingdom"] != ood_kingdom].copy()

    print(f"OOD ({ood_kingdom}): {len(ood_df)}")
    print(f"Main: {len(main_df)}")

    if "cluster_id" in main_df.columns:
        clusters = main_df["cluster_id"].unique()
        rng.shuffle(clusters)
        n  = len(clusters)
        n1 = int(train_frac * n)
        n2 = int(val_frac   * n)
        tc = set(clusters[:n1])
        vc = set(clusters[n1:n1+n2])
        ec = set(clusters[n1+n2:])
        train_df = main_df[main_df["cluster_id"].isin(tc)]
        val_df   = main_df[main_df["cluster_id"].isin(vc)]
        test_df  = main_df[main_df["cluster_id"].isin(ec)]
    else:
        idx = rng.permutation(len(main_df))
        n   = len(idx)
        n1  = int(train_frac * n)
        n2  = int(val_frac   * n)
        train_df = main_df.iloc[idx[:n1]]
        val_df   = main_df.iloc[idx[n1:n1+n2]]
        test_df  = main_df.iloc[idx[n1+n2:]]

    print(f"Train:{len(train_df)} Val:{len(val_df)} "
          f"Test:{len(test_df)} OOD:{len(ood_df)}")
    return {"train": train_df, "val": val_df,
            "test": test_df,   "ood": ood_df}


# ─────────────────────────────────────────────
# MODEL
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
    """
    One-hot sequence → mean-pooled embedding.
    Input:  (B, L, 20)
    Output: (B, d_model)
    """
    def __init__(self, d_model=256, n_heads=8,
                 n_layers=4, ffn_dim=512,
                 dropout=0.1, max_len=512):
        super().__init__()
        self.proj   = nn.Linear(VOCAB_SIZE, d_model)
        self.pe     = PositionalEncoding(d_model, max_len, dropout)
        layer       = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=ffn_dim, dropout=dropout,
            batch_first=True, norm_first=True,
            activation="gelu",
        )
        self.enc    = nn.TransformerEncoder(
            layer, num_layers=n_layers,
            enable_nested_tensor=False)
        self.norm   = nn.LayerNorm(d_model)

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


class KESTREL(nn.Module):
    """
    Sequence + kingdom label → 10 IDP ensemble properties.
    Kingdom embedding is concatenated to the encoder output before both
    prediction heads. Unknown kingdoms use a zero embedding (OOD-safe).
    """
    def __init__(self, d_model=256, n_heads=8,
                 n_layers=4, ffn_dim=512,
                 dropout=0.1, max_len=256
                 ,
                 n_kingdoms=12, kingdom_dim=KINGDOM_DIM):
        super().__init__()
        self.encoder    = SequenceEncoder(
            d_model, n_heads, n_layers, ffn_dim, dropout, max_len)

        # +1 for the zero-vector fallback (index = n_kingdoms)
        # padding_idx ensures the fallback row is always zero and
        # receives no gradient — clean separation from trained embeddings
        self.kingdom_emb = nn.Embedding(
            n_kingdoms + 1, kingdom_dim,
            padding_idx=n_kingdoms)

        head_in = d_model + kingdom_dim
        self.geo_head   = PredHead(head_in, 128, len(GEO_TARGETS),   dropout)
        self.graph_head = PredHead(head_in, 128, len(GRAPH_TARGETS), dropout)

    def _kingdom_vec(self, kingdom_idx):
        """
        Map kingdom indices to embeddings.
        Index -1 (unknown/OOD) → padding index → zero vector.
        """
        # replace -1 with padding_idx (n_kingdoms)
        safe_idx = kingdom_idx.clone()
        safe_idx[safe_idx < 0] = self.kingdom_emb.padding_idx
        return self.kingdom_emb(safe_idx)   # (B, kingdom_dim)

    def forward(self, seq, mask, kingdom_idx):
        emb  = self.encoder(seq, mask)                    # (B, d_model)
        kvec = self._kingdom_vec(kingdom_idx)              # (B, kingdom_dim)
        h    = torch.cat([emb, kvec], dim=-1)             # (B, d_model+kingdom_dim)
        return torch.cat([self.geo_head(h),
                          self.graph_head(h)], dim=-1)    # (B, 10)

    def n_params(self):
        return sum(p.numel() for p in self.parameters()
                   if p.requires_grad)

    @torch.no_grad()
    def predict(self, sequences, target_stats, device,
                kingdoms=None, kingdom_vocab=None,
                max_len=256, batch_size=128):
        """
        Parameters
        ----------
        sequences     : list of amino acid strings
        target_stats  : {"mean": Series, "std": Series}
        kingdoms      : optional list of kingdom strings, same length as
                        sequences. Unknown/missing → zero embedding.
        kingdom_vocab : dict kingdom_name → int index (from training).
        """
        self.eval()
        vocab  = kingdom_vocab or {}
        if kingdoms is None:
            kingdoms = ["Unknown"] * len(sequences)

        preds = []
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i+batch_size]
            batch_ks   = kingdoms[i:i+batch_size]
            B = len(batch_seqs)
            oh   = torch.zeros(B, max_len, VOCAB_SIZE)
            mask = torch.zeros(B, max_len, dtype=torch.bool)
            kidx = torch.full((B,), -1, dtype=torch.long)
            for b, (seq, k) in enumerate(zip(batch_seqs, batch_ks)):
                seq = seq[:max_len]
                for j, aa in enumerate(seq):
                    idx = AA_TO_IDX.get(aa, -1)
                    if idx >= 0:
                        oh[b, j, idx] = 1.0
                mask[b, :len(seq)] = True
                kidx[b] = vocab.get(k, -1)
            p = self(oh.to(device), mask.to(device),
                     kidx.to(device)).cpu()
            mean = torch.tensor(target_stats["mean"].values,
                                dtype=torch.float32)
            std  = torch.tensor(target_stats["std"].values,
                                dtype=torch.float32)
            preds.append(p * std + mean)
        return pd.DataFrame(
            torch.cat(preds).numpy(), columns=ALL_TARGETS)


class PhyschemMLP(nn.Module):
    """
    Ablation: 13 scalar sequence features → 10 targets.
    No sequence information beyond simple statistics.
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


def r2_by_kingdom(preds, targets, kingdoms):
    kingdoms = np.array(kingdoms)
    rows = []
    for k in sorted(set(kingdoms)):
        idx = np.where(kingdoms == k)[0]
        r2  = r2_scores(preds[idx], targets[idx])
        r2["kingdom"] = k
        r2["n"]       = len(idx)
        rows.append(r2)
    return pd.DataFrame(rows).set_index("kingdom")


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
    """
    Parameters
    ----------
    patience      : early-stop patience on smoothed val loss.
                    Default 15 for KESTREL, use 20 for PhyschemMLP.
    smooth_window : epochs to average for early-stop decisions.
    """
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
          f"{'Rg':>7} {'nu':>7} {'Δ':>7} {'geff':>7}")
    print("─" * 55)

    for ep in range(epochs):
        # train
        model.train()
        tl = 0
        for b in train_loader:
            loss, _, _ = step(model, b, device, is_kestrel)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tl += loss.item()
        tl /= len(train_loader)

        # validate
        model.eval()
        vl, vp, vt, _ = evaluate(model, val_loader, device, is_kestrel)
        sched.step()
        r2 = r2_scores(vp, vt)
        hist.append({"epoch": ep+1, "train": tl,
                      "val": vl, **r2})

        print(f"{ep+1:>4} {tl:>8.4f} {vl:>8.4f} "
              f"{r2['rg']:>7.3f} {r2['nu']:>7.3f} "
              f"{r2['delta']:>7.3f} "
              f"{r2['global_efficiency']:>7.3f}")

        # smoothed early stopping
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


def full_eval(model, test_loader, ood_loader,
              out_dir, name, device, is_kestrel=True):
    print(f"\n{'='*55}\nEvaluation: {name}\n{'='*55}")
    for split, loader in [("test", test_loader),
                           ("ood",  ood_loader)]:
        _, p, t, k = evaluate(model, loader, device, is_kestrel)
        r2 = r2_scores(p, t)
        kr = r2_by_kingdom(p, t, k)

        print(f"\n── {split.upper()} ──────────────────────────────")
        for tgt, score in r2.items():
            print(f"  {tgt:30s}: {score:.4f}")
        print(f"\n── {split.upper()} by kingdom ──────────────────")
        print(kr[["rg","nu","delta","global_efficiency","n"]])

        pd.DataFrame([{"model":name,"split":split,**r2}]).to_csv(
            os.path.join(out_dir, f"{name}_{split}_r2.csv"),
            index=False)
        kr.to_csv(
            os.path.join(out_dir,
                         f"{name}_{split}_kingdom_r2.csv"))


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run(data_csv, out_dir="kestrel_output/", max_len=256,
        batch_size=256, epochs=100, patience=15,
        lr=5e-4, seed=42):

    torch.manual_seed(seed); np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    device = get_device()

    # load
    df = pd.read_csv(data_csv)
    df = df.dropna(subset=["rg","ree","nu","sequence"])
    for col in GRAPH_TARGETS:
        if col not in df.columns:
            df[col] = np.nan
    print(f"Loaded {len(df)} sequences | "
          f"{df['kingdom'].nunique()} kingdoms")

    # split
    splits = make_splits(df, seed=seed)
    stats  = {
        "mean": splits["train"][ALL_TARGETS].mean(),
        "std":  splits["train"][ALL_TARGETS].std().clip(lower=1e-6),
    }

    # kingdom vocab from training split only
    kingdom_vocab = build_kingdom_vocab(splits["train"])
    n_kingdoms    = len(kingdom_vocab)
    print(f"Kingdom vocab ({n_kingdoms}): {', '.join(sorted(kingdom_vocab))}")
    save_kingdom_vocab(kingdom_vocab,
                       os.path.join(out_dir, "kingdom_vocab.txt"))

    # datasets
    kw = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    def make_loader(split_df, shuffle):
        ds = IDPDataset(split_df, max_len, stats, kingdom_vocab)
        return DataLoader(ds, shuffle=shuffle, **kw)

    train_dl = make_loader(splits["train"], True)
    val_dl   = make_loader(splits["val"],   False)
    test_dl  = make_loader(splits["test"],  False)
    ood_dl   = make_loader(splits["ood"],   False)

    # ── KESTREL ───────────────────────────────────────────────
    kestrel = train_model(
        KESTREL(n_kingdoms=n_kingdoms, kingdom_dim=KINGDOM_DIM),
        train_dl, val_dl, out_dir,
        "kestrel", epochs, patience, lr,
        device=device, is_kestrel=True, smooth_window=3)
    full_eval(kestrel, test_dl, ood_dl,
              out_dir, "kestrel", device)

    # save normalization stats
    stats["mean"].to_csv(os.path.join(out_dir, "target_mean.csv"))
    stats["std"].to_csv( os.path.join(out_dir, "target_std.csv"))

    # ── PhyschemMLP ablation ──────────────────────────────────
    if all(c in df.columns for c in PHYSCHEM_COLS):
        pc_model = train_model(
            PhyschemMLP(), train_dl, val_dl, out_dir,
            "physchem_mlp", epochs, patience=20, lr=lr,
            device=device, is_kestrel=False, smooth_window=3)
        full_eval(pc_model, test_dl, ood_dl,
                  out_dir, "physchem_mlp", device,
                  is_kestrel=False)

    # ── final summary ─────────────────────────────────────────
    print(f"\n{'='*55}")
    print("SUMMARY TABLE (Test set)")
    print(f"{'Model':20s} {'Rg':>7} {'Ree':>7} {'nu':>7} "
          f"{'Δ':>7} {'geff':>7}")
    for name, fname in [("KESTREL",     "kestrel_test_r2.csv"),
                         ("PhyschemMLP", "physchem_mlp_test_r2.csv")]:
        fp = os.path.join(out_dir, fname)
        if os.path.exists(fp):
            r = pd.read_csv(fp).iloc[0]
            print(f"{name:20s} {r['rg']:>7.3f} {r['ree']:>7.3f} "
                  f"{r['nu']:>7.3f} {r['delta']:>7.3f} "
                  f"{r['global_efficiency']:>7.3f}")

    return kestrel


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data",       required=True)
    p.add_argument("--out",        default="kestrel_output/")
    p.add_argument("--max_len",    default=256,  type=int)
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
# Step 1 — merge your CSV with trajectory output:
#
#   import pandas as pd
#   idp  = pd.read_csv("idp_dataset.csv")
#   traj = pd.read_csv("results/scalar_features.csv")
#   df   = idp.merge(traj, left_on="uniprot_id",
#                    right_on="protein_name", how="inner")
#   df.to_csv("merged.csv", index=False)
#
# Step 2 — train:
#
#   python kestrel.py --data merged.csv --out kestrel_output/
#
# Step 3 — inference on new sequences (with kingdom label):
#
#   from kestrel import KESTREL, load_kingdom_vocab
#   import torch, pandas as pd
#
#   vocab = load_kingdom_vocab("kestrel_output/kingdom_vocab.txt")
#   model = KESTREL(n_kingdoms=len(vocab))
#   model.load_state_dict(
#       torch.load("kestrel_output/kestrel_best.pt"))
#
#   stats = {
#       "mean": pd.read_csv("kestrel_output/target_mean.csv",
#                           index_col=0).squeeze(),
#       "std":  pd.read_csv("kestrel_output/target_std.csv",
#                           index_col=0).squeeze(),
#   }
#   seqs     = ["MAEADFKMVSEPVAHGVAEE", "MGKGTDMARAKARRLKGMK"]
#   kingdoms = ["Bacteria", "Fungi"]   # or None for zero-embedding fallback
#   preds = model.predict(seqs, stats, torch.device("cpu"),
#                         kingdoms=kingdoms, kingdom_vocab=vocab)
#   print(preds)
#
# ─────────────────────────────────────────────
