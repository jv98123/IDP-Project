# BENDER & KESTREL

**BENDER** (Benchmark for Ensemble and Network-based Disordered protein Ensemble Representation) is the first cross-kingdom IDP ensemble dataset, comprising 11,533 intrinsically disordered protein sequences across 13 taxonomic kingdoms with CALVADOS-2 conformational ensembles and ten ensemble-level prediction targets — five geometric properties and five novel ensemble contact network properties.

**KESTREL** (Kingdom-aware Ensemble Sequence Transformer for Representing Ensemble-Level properties) is a lightweight transformer (~2.2M parameters) trained from scratch on one-hot encoded sequences that predicts all ten targets across kingdoms, including held-out Viruses and Archaea used as out-of-distribution (OOD) test kingdoms.

---

## Requirements

```bash
pip install torch numpy pandas tqdm transformers huggingface_hub
```

A CUDA-capable GPU is strongly recommended for all experiments.

---

## Dataset

BENDER is hosted on Hugging Face: `huggingface.co/datasets/taseef/BENDER`

The dataset CSV (`bender_v1.csv`) must contain the following columns:

| Column | Description |
|--------|-------------|
| `sequence` | Amino acid sequence |
| `kingdom` | Taxonomic kingdom (e.g. Bacteria, Fungi, Viruses, Archaea) |
| `rg` | Radius of gyration (nm) |
| `ree` | End-to-end distance (nm) |
| `nu` | Flory scaling exponent |
| `delta` | Asphericity |
| `A0` | Flory prefactor |
| `global_efficiency` | Ensemble contact network global efficiency |
| `fragmentation_index` | Fraction of residues in largest connected component |
| `avg_clustering` | Average clustering coefficient |
| `transitivity` | Network transitivity |
| `degree_assortativity` | Degree assortativity |

---

## Repository Structure

```
kestrel_ood_virus.py        # KESTREL — OOD Viruses experiment
kestrel_ood_archaea.py      # KESTREL — OOD Archaea experiment
kestrel_human.py            # KESTREL-Human — IDRome in-distribution experiment
idp_esm2_virus.py           # IDP-ESM2 — OOD Viruses experiment
idp_esm_archaea.py          # IDP-ESM2 — OOD Archaea experiment
run_albatross.py            # ALBATROSS evaluation
traj_analysis_fast.py       # Trajectory analysis pipeline
```

---

## Experiments

### 1. KESTREL — OOD Viruses

Trains KESTREL on BENDER with Viruses held out as the OOD kingdom.

```bash
# Basic run
python kestrel_ood_virus.py --data bender_v1.csv --out kestrel_virus_seed42/

# Reproducing paper results (3 seeds)
python kestrel_ood_virus.py --data bender_v1.csv --out kestrel_virus_seed42/ --seed 42
python kestrel_ood_virus.py --data bender_v1.csv --out kestrel_virus_seed67/ --seed 67
python kestrel_ood_virus.py --data bender_v1.csv --out kestrel_virus_seed93/ --seed 93
```

**Outputs**

| File | Description |
|------|-------------|
| `kestrel_best.pt` | Best model checkpoint |
| `kingdom_vocab.txt` | Kingdom index mapping |
| `kestrel_test_r2.csv` | R² on held-out test set |
| `kestrel_ood_r2.csv` | R² on OOD Virus sequences |
| `kestrel_test_kingdom_r2.csv` | Per-kingdom R² on test set |
| `kestrel_ood_kingdom_r2.csv` | Per-kingdom R² on OOD set |
| `physchem_mlp_test_r2.csv` | PhyschemMLP R² on test set |
| `physchem_mlp_ood_r2.csv` | PhyschemMLP R² on OOD set |

**Loading a checkpoint**

```python
import torch
from kestrel_ood_virus import KESTREL, load_kingdom_vocab

vocab = load_kingdom_vocab("kestrel_virus_seed42/kingdom_vocab.txt")
model = KESTREL(n_kingdoms=len(vocab))
model.load_state_dict(
    torch.load("kestrel_virus_seed42/kestrel_best.pt", weights_only=False))
model.eval()
```

---

### 2. KESTREL — OOD Archaea

Trains KESTREL on BENDER with Archaea held out as the OOD kingdom.

```bash
# Basic run
python kestrel_ood_archaea.py --data bender_v1.csv --out kestrel_archaea_seed42/

# Reproducing paper results (3 seeds)
python kestrel_ood_archaea.py --data bender_v1.csv --out kestrel_archaea_seed42/ --seed 42
python kestrel_ood_archaea.py --data bender_v1.csv --out kestrel_archaea_seed67/ --seed 67
python kestrel_ood_archaea.py --data bender_v1.csv --out kestrel_archaea_seed93/ --seed 93
```

**Outputs** — same structure as OOD Viruses, with `kestrel_ood_r2.csv` containing Archaea OOD results.

**Loading a checkpoint**

```python
import torch
from kestrel_ood_archaea import KESTREL, load_kingdom_vocab

vocab = load_kingdom_vocab("kestrel_archaea_seed42/kingdom_vocab.txt")
model = KESTREL(n_kingdoms=len(vocab))
model.load_state_dict(
    torch.load("kestrel_archaea_seed42/kestrel_best.pt", weights_only=False))
model.eval()
```

---

### 3. KESTREL-Human — IDRome In-Distribution

Architecture-matched KESTREL variant trained on Human-IDRome without the kingdom embedding. Used for controlled comparison against GeoGraph on the same data and evaluation protocol.

**Input:** `idrome_clustered.csv` with columns `fasta`, `Rg/nm`, `Ree/nm`, `nu`, `Delta`.

```bash
# Basic run
python kestrel_human.py --data idrome_clustered.csv --out kestrel_human_seed42/

# Reproducing paper results (3 seeds)
python kestrel_human.py --data idrome_clustered.csv --out kestrel_human_seed42/ --seed 42
python kestrel_human.py --data idrome_clustered.csv --out kestrel_human_seed67/ --seed 67
python kestrel_human.py --data idrome_clustered.csv --out kestrel_human_seed93/ --seed 93
```

**Outputs**

| File | Description |
|------|-------------|
| `kestrel_human_best.pt` | Best model checkpoint |
| `kestrel_human_test_r2.csv` | R² on IDRome test set |
| `physchem_mlp_test_r2.csv` | PhyschemMLP R² on IDRome test set |

**Loading a checkpoint**

```python
import torch
from kestrel_human import KESTREL

model = KESTREL()
model.load_state_dict(
    torch.load("kestrel_human_seed42/kestrel_human_best.pt", weights_only=False))
model.eval()
```

---

### 4. IDP-ESM2 — OOD Viruses

Evaluates frozen IDP-ESM2-8M and IDP-ESM2-150M backbones with a GeoHead MLP trained on BENDER, evaluated on OOD Viruses. ESM2 model weights are downloaded automatically from HuggingFace on first run.

```bash
# Run both 8M and 150M
python idp_esm2_virus.py --bender bender_v1.csv --out esm_virus_seed42/

# Reproducing paper results (3 seeds)
python idp_esm2_virus.py --bender bender_v1.csv --out esm_virus_8M_seed42/  --models 8M   --seed 42
python idp_esm2_virus.py --bender bender_v1.csv --out esm_virus_8M_seed67/  --models 8M   --seed 67
python idp_esm2_virus.py --bender bender_v1.csv --out esm_virus_8M_seed93/  --models 8M   --seed 93
python idp_esm2_virus.py --bender bender_v1.csv --out esm_virus_150M_seed42/ --models 150M --seed 42
python idp_esm2_virus.py --bender bender_v1.csv --out esm_virus_150M_seed67/ --models 150M --seed 67
python idp_esm2_virus.py --bender bender_v1.csv --out esm_virus_150M_seed93/ --models 150M --seed 93
```

**Outputs**

| File | Description |
|------|-------------|
| `IDP-ESM2-{size}_geohead_best.pt` | Best GeoHead checkpoint |
| `all_results.csv` | R² for BENDER test and OOD Viruses |

**Loading a checkpoint**

```python
import torch
from idp_esm2_virus import GeoHead

model = GeoHead(hidden_dim=320)  # 320 for 8M, 640 for 150M
ckpt  = torch.load("esm_virus_8M_seed42/IDP-ESM2-8M_geohead_best.pt",
                   weights_only=False)
model.load_state_dict(ckpt["state_dict"])
model.eval()
```

---

### 5. IDP-ESM2 — OOD Archaea

Identical protocol to OOD Viruses with Archaea held out instead.

```bash
# Run both 8M and 150M
python idp_esm_archaea.py --bender bender_v1.csv --out esm_archaea_seed42/

# Reproducing paper results (3 seeds)
python idp_esm_archaea.py --bender bender_v1.csv --out esm_archaea_8M_seed42/  --models 8M   --seed 42
python idp_esm_archaea.py --bender bender_v1.csv --out esm_archaea_8M_seed67/  --models 8M   --seed 67
python idp_esm_archaea.py --bender bender_v1.csv --out esm_archaea_8M_seed93/  --models 8M   --seed 93
python idp_esm_archaea.py --bender bender_v1.csv --out esm_archaea_150M_seed42/ --models 150M --seed 42
python idp_esm_archaea.py --bender bender_v1.csv --out esm_archaea_150M_seed67/ --models 150M --seed 67
python idp_esm_archaea.py --bender bender_v1.csv --out esm_archaea_150M_seed93/ --models 150M --seed 93
```

**Loading a checkpoint**

```python
import torch
from idp_esm_archaea import GeoHead

model = GeoHead(hidden_dim=320)  # 320 for 8M, 640 for 150M
ckpt  = torch.load("esm_archaea_8M_seed42/IDP-ESM2-8M_geohead_best.pt",
                   weights_only=False)
model.load_state_dict(ckpt["state_dict"])
model.eval()
```

---

## Arguments (all scripts)

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` / `--bender` | required | Path to input CSV |
| `--out` | script-dependent | Output directory |
| `--seed` | `42` | Random seed |
| `--epochs` | `100` | Maximum training epochs |
| `--patience` | `15` | Early stopping patience |
| `--lr` | `5e-4` | Learning rate |
| `--batch_size` | `256` | Training batch size |
| `--max_len` | `256` | Maximum sequence length |
| `--models` | `8M 150M` | IDP-ESM2 only: model size(s) to run |
| `--device` | auto | Force device (`cuda` or `cpu`) |

---

## Citation

If you use BENDER or KESTREL in your work, please cite:

```bibtex
@inproceedings{bender_kestrel_2025,
    title     = {BENDER: A Cross-Kingdom Benchmark for Intrinsically
                 Disordered Protein Ensemble Prediction},
    booktitle = {NeurIPS Datasets and Benchmarks Track},
    year      = {2025},
}
```
