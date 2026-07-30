"""Central configuration for PepD — peptide property prediction (inference only).

Before running, set the following environment variables or edit the paths below:
    - PEPD_CKPT_ROOT: root directory containing per-property fold_*.pt checkpoints
    - ESM_MODEL_PATH: path to facebook/esm2_t33_650M_UR50D model directory
"""

import os

# ─── Path Configuration ──────────────────────────────────────────────
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CODE_DIR)

# PepD property checkpoint root — fold_{1..5}.pt per property live here.
# Expected layout: <PEPD_CKPT_ROOT>/<property>/fold_{1..5}.pt
PEPD_CKPT_ROOT = os.environ.get(
    "PEPD_CKPT_ROOT",
    os.path.join(PROJECT_DIR, "model_weights", "PepD"),
)

# ESM2-650M model for per-residue embedding extraction.
ESM_MODEL_PATH = os.environ.get(
    "ESM_MODEL_PATH",
    os.path.join(PROJECT_DIR, "model_weights", "ESM-2"),
)

# iupred2a disorder prediction scripts and data.
IUPRED_DIR = os.path.join(CODE_DIR, "iupred")