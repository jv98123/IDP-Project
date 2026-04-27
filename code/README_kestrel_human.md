# KESTREL-Human — IDRome Experiment

KESTREL-Human is a variant of KESTREL trained exclusively on the **IDRome dataset** of human IDPs. Unlike the cross-kingdom KESTREL variants, this model does not use kingdom embeddings and is evaluated purely on in-distribution IDRome sequences.

---

## Requirements

```bash
pip install torch numpy pandas tqdm
```

A CUDA-capable GPU is strongly recommended.

---

## Input Data

| File | Description |
|------|-------------|
| `idrome_clustered.csv` | IDRome dataset of human IDP sequences with geometric targets |

The CSV must contain the following columns:
- `fasta` — amino acid sequence
- `Rg/nm` — radius of gyration
- `Ree/nm` — end-to-end distance
- `nu` — scaling exponent
- `Delta` — asphericity

---

## Usage

### Basic run (seed 42)
```bash
python kestrel_human.py --data idrome_clustered.csv --out kestrel_human_seed42/
```

### Run with all options
```bash
python kestrel_human.py \
    --data       idrome_clustered.csv \
    --out        kestrel_human_seed42/ \
    --seed       42 \
    --epochs     100 \
    --patience   15 \
    --lr         5e-4 \
    --batch_size 256 \
    --max_len    256
```

### Reproducing paper results (3 seeds)
```bash
python kestrel_human.py --data idrome_clustered.csv --out kestrel_human_seed42/ --seed 42
python kestrel_human.py --data idrome_clustered.csv --out kestrel_human_seed67/ --seed 67
python kestrel_human.py --data idrome_clustered.csv --out kestrel_human_seed93/ --seed 93
```

---

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | required | Path to IDRome CSV |
| `--out` | `kestrel_human_output/` | Output directory |
| `--seed` | `42` | Random seed |
| `--epochs` | `100` | Maximum training epochs |
| `--patience` | `15` | Early stopping patience |
| `--lr` | `5e-4` | Learning rate |
| `--batch_size` | `256` | Training batch size |
| `--max_len` | `256` | Maximum sequence length (must match GeoGraph for fair comparison) |

---

## Outputs

All outputs are saved to the `--out` directory:

| File | Description |
|------|-------------|
| `kestrel_human_best.pt` | Best model checkpoint |
| `target_mean.csv` | Target normalization means |
| `target_std.csv` | Target normalization standard deviations |
| `kestrel_human_test_r2.csv` | R² scores on IDRome test set |
| `physchem_mlp_best.pt` | Best PhyschemMLP checkpoint |
| `physchem_mlp_test_r2.csv` | PhyschemMLP R² on test set |

---

## Loading a Pretrained Checkpoint

```python
import torch
from kestrel_human import KESTREL

model = KESTREL()
model.load_state_dict(
    torch.load("kestrel_human_seed42/kestrel_human_best.pt", weights_only=False))
model.eval()
```
