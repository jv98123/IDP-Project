# KESTREL — OOD Viruses Experiment

KESTREL (Kingdom-Embedded Sequence Transformer for IDP Property Prediction) is a transformer-based model that predicts geometric and graph-theoretic properties of intrinsically disordered proteins (IDPs) directly from amino acid sequence.

This script trains KESTREL on the BENDER dataset with **Viruses held out as the OOD kingdom**, then evaluates on both a held-out test set and the Virus OOD sequences.

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
| `bender_v1.csv` | BENDER dataset containing IDP sequences, kingdom labels, and target properties |

The CSV must contain the following columns:
- `sequence` — amino acid sequence
- `kingdom` — taxonomic kingdom (e.g. Bacteria, Fungi, Viruses)
- `rg`, `ree`, `nu`, `delta`, `A0` — geometric targets
- `global_efficiency`, `fragmentation_index`, `avg_clustering`, `transitivity`, `degree_assortativity` — graph targets

---

## Usage

### Basic run (seed 42)
```bash
python kestrel_ood_virus.py --data bender_v1.csv --out kestrel_virus_seed42/
```

### Run with all options
```bash
python kestrel_ood_virus.py \
    --data       bender_v1.csv \
    --out        kestrel_virus_seed42/ \
    --seed       42 \
    --epochs     100 \
    --patience   15 \
    --lr         5e-4 \
    --batch_size 256 \
    --max_len    256
```

### Reproducing paper results (3 seeds)
```bash
python kestrel_ood_virus.py --data bender_v1.csv --out kestrel_virus_seed42/ --seed 42
python kestrel_ood_virus.py --data bender_v1.csv --out kestrel_virus_seed67/ --seed 67
python kestrel_ood_virus.py --data bender_v1.csv --out kestrel_virus_seed93/ --seed 93
```

---

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | required | Path to BENDER CSV |
| `--out` | `kestrel_output/` | Output directory |
| `--seed` | `42` | Random seed |
| `--epochs` | `100` | Maximum training epochs |
| `--patience` | `15` | Early stopping patience |
| `--lr` | `5e-4` | Learning rate |
| `--batch_size` | `256` | Training batch size |
| `--max_len` | `256` | Maximum sequence length |

---

## Outputs

All outputs are saved to the `--out` directory:

| File | Description |
|------|-------------|
| `kestrel_best.pt` | Best model checkpoint |
| `target_mean.csv` | Target normalization means |
| `target_std.csv` | Target normalization standard deviations |
| `kingdom_vocab.txt` | Kingdom index mapping |
| `kestrel_test_r2.csv` | R² scores on held-out test set |
| `kestrel_ood_r2.csv` | R² scores on OOD Virus sequences |
| `kestrel_test_kingdom_r2.csv` | Per-kingdom R² on test set |
| `kestrel_ood_kingdom_r2.csv` | Per-kingdom R² on OOD set |
| `physchem_mlp_best.pt` | Best PhyschemMLP checkpoint |
| `physchem_mlp_test_r2.csv` | PhyschemMLP R² on test set |
| `physchem_mlp_ood_r2.csv` | PhyschemMLP R² on OOD set |

---

## Loading a Pretrained Checkpoint

```python
import torch
import pandas as pd
from kestrel_ood_virus import KESTREL, load_kingdom_vocab

vocab = load_kingdom_vocab("kestrel_virus_seed42/kingdom_vocab.txt")
model = KESTREL(n_kingdoms=len(vocab))
model.load_state_dict(
    torch.load("kestrel_virus_seed42/kestrel_best.pt", weights_only=False))
model.eval()
```
