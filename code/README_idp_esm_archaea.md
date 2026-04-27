# IDP-ESM2 — OOD Archaea Experiment

This script evaluates the pretrained **IDP-ESM2-8M** and **IDP-ESM2-150M** models on the BENDER dataset with **Archaea as the OOD kingdom**. A lightweight GeoHead MLP is trained on top of frozen ESM2 embeddings to predict IDP geometric properties, then evaluated on a BENDER test set and OOD Archaea sequences.

---

## Requirements

```bash
pip install torch transformers huggingface_hub numpy pandas tqdm
```

A CUDA-capable GPU is strongly recommended. The ESM2 model weights will be downloaded automatically from HuggingFace on first run.

---

## Input Data

| File | Description |
|------|-------------|
| `bender_v1.csv` | BENDER dataset containing IDP sequences, kingdom labels, and target properties |

The CSV must contain the following columns:
- `sequence` — amino acid sequence
- `kingdom` — taxonomic kingdom
- `rg`, `ree`, `nu`, `delta`, `a0` — geometric targets

---

## Usage

### Run both 8M and 150M (seed 42)
```bash
python idp_esm_archaea.py --bender bender_v1.csv --out esm_archaea_seed42/
```

### Run a single model size
```bash
python idp_esm_archaea.py --bender bender_v1.csv --out esm_archaea_8M_seed42/ --models 8M --seed 42
python idp_esm_archaea.py --bender bender_v1.csv --out esm_archaea_150M_seed42/ --models 150M --seed 42
```

### Reproducing paper results (3 seeds)
```bash
python idp_esm_archaea.py --bender bender_v1.csv --out esm_archaea_8M_seed42/ --models 8M --seed 42
python idp_esm_archaea.py --bender bender_v1.csv --out esm_archaea_8M_seed67/ --models 8M --seed 67
python idp_esm_archaea.py --bender bender_v1.csv --out esm_archaea_8M_seed93/ --models 8M --seed 93

python idp_esm_archaea.py --bender bender_v1.csv --out esm_archaea_150M_seed42/ --models 150M --seed 42
python idp_esm_archaea.py --bender bender_v1.csv --out esm_archaea_150M_seed67/ --models 150M --seed 67
python idp_esm_archaea.py --bender bender_v1.csv --out esm_archaea_150M_seed93/ --models 150M --seed 93
```

---

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--bender` | required | Path to BENDER CSV |
| `--out` | `idp_esm2_results/` | Output directory |
| `--models` | `8M 150M` | Which model(s) to run (`8M`, `150M`, or both) |
| `--seed` | `42` | Random seed |
| `--device` | auto | Force device (`cuda` or `cpu`) |

---

## Outputs

All outputs are saved to the `--out` directory:

| File | Description |
|------|-------------|
| `IDP-ESM2-{size}_geohead_best.pt` | Best GeoHead checkpoint |
| `all_results.csv` | R² scores for all splits (BENDER test + OOD Archaea) |

---

## Loading a Pretrained GeoHead Checkpoint

```python
import torch
from idp_esm_archaea import GeoHead

model = GeoHead(hidden_dim=320)  # 320 for 8M, 640 for 150M
ckpt  = torch.load("esm_archaea_8M_seed42/IDP-ESM2-8M_geohead_best.pt",
                   weights_only=False)
model.load_state_dict(ckpt["state_dict"])
model.eval()
```
