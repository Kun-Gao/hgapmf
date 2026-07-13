# HG-APMF Model

This is the official PyTorch implementation of the model:

**HG-APMF: HyperGraph-guided Anatomical Prototype Modality Fusion**

## What Is Included

- `hgapmf/model.py`
  - `HGAPMF_Net`
  - `HGAPMFBlock`
  - stage-specific shared anatomical prototype retrieval
  - prototype-centered hypergraph reasoning
  - refinement-aware fusion
  - progressive outputs `S2/S4/S8/S14`
  - `ct_mr`, `ct_only`, and `mr_only` forward modes
- `hgapmf/spatial_refinement_cue.py`
  - `SpatialRefinementCuePredictor`
- `smoke_test.py`

## Dependency

Only PyTorch is required.

```bash
pip install -r requirements.txt
```

## Quick Start

```python
import torch
from hgapmf import EXP018Config, HGAPMF_EXP018_Net

model = HGAPMF_EXP018_Net(
    in_channels=2,
    num_classes=15,
    base_channels=16,
    exp018_config=EXP018Config(
        prototype_dim=32,
        prototype_nums=(8, 16, 32),
        max_tokens=(64, 64, 128),
        hypergraph_topk=4,
        num_heads=4,
    ),
)

x = torch.randn(1, 2, 128, 128, 128)  # channel 0 = CT, channel 1 = MR
logits = model(x, mode="ct_mr")       # [B, 15, D, H, W]
debug = model(x, return_debug=True, mode="ct_mr")
```

## Forward Modes

- `mode="ct_mr"`: complete CT + MR input.
- `mode="ct_only"`: CT-only inference; the MR branch receives a zero placeholder internally.
- `mode="mr_only"`: MR-only inference; the CT branch receives a zero placeholder internally.

## Progressive Outputs

The model returns four semantic stages when `return_debug=True`:

| Stage | Channels | Meaning |
|---|---:|---|
| `S2` | 3 | background + 2 foreground groups |
| `S4` | 5 | background + 4 foreground groups |
| `S8` | 9 | background + 8 foreground groups |
| `S14` | 15 | background + 14 anatomical classes |

Raw labels are assumed to be:

- `0`: background
- `1..14`: foreground anatomical classes

The progressive label maps are exposed as `PROGRESSIVE_LABEL_MAPS`.

## Smoke Test

CPU:

```bash
python smoke_test.py --device cpu --shape 64 64 64 --base-channels 4 --prototype-dim 16
```

CUDA:

```bash
python smoke_test.py --device cuda --shape 64 64 64 --base-channels 4 --prototype-dim 16
```

## Notes For Training Integration

This folder does not include the training code. The complete implementation will be made publicly available upon acceptance of the paper:

- segmentation loss for `S2/S4/S8/S14`
- progressive target mapping using `PROGRESSIVE_LABEL_MAPS`
- refinement cue supervision
- complete-to-single contrastive alignment loss
- prototype usage/diversity regularization