"""
predict.py — End-to-end inference: peptide sequence → 9 property desirability scores.

Loads the 5-fold ensemble of PTNet checkpoints for each of the 9 properties,
runs full feature extraction (ESM2 + iupred2a + RDKit), and outputs per-property
scores in one pass.

Output convention (uniform across all 9 properties):
    d_<property> ∈ [0, 1], higher = better.

For classification tasks (allergen, hemolysis, nonfouling, permeability,
solubility, toxicity): d is the predicted probability of the positive class
(1 = non-toxic, non-allergenic, etc. — direction aligned during training).

For regression tasks (halflife_invitro, halflife_invivo, stability):
d is the sigmoid-normalized score from BCEWithLogitsLoss training, already in [0,1].

■■■ Usage ■■■

    # Single peptide
    python predict.py --sequence "GSPLRYQWTIDYKGR" --gpu 0

    # Batch from TSV/CSV (columns: name, sequence)
    python predict.py --input candidates.tsv --output scored.tsv --gpu 0

    # Use a custom checkpoint directory
    python predict.py --sequence "..." --ckpt_root /path/to/PepD --gpu 0

■■■ Speed ■■■

    First call loads ESM2 (650M params) + 9 properties × 5 folds = 45 PTNet
    checkpoints (~1.7 GB GPU RAM). ~3-5 minutes cold start. After warmup each
    new peptide takes ~5-10 seconds (dominated by iupred2a CLI subprocess).
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Ensure local modules are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from amnet.net import AMNet
from smnet.net import SMNet
from net import PTNet
from amnet.pep_graph import featurize_peptide_graph
from smnet.smiles_graph import featurize_smiles_graph
from config import ESM_MODEL_PATH, IUPRED_DIR, PEPD_CKPT_ROOT


# ─────────────────────────────────────────────────────────────────────
# 9 properties (no immunogenicity)
# ─────────────────────────────────────────────────────────────────────

# These must match the dataset names in config.py and output/<name>/ dirs.
PROPERTIES = [
    "allergen",
    "halflife_invitro",
    "halflife_invivo",
    "hemolysis",
    "nonfouling",
    "permeability",
    "solubility",
    "stability",
    "toxicity",
]

# Task type per property (used only for display; models output sigmoid[0,1])
PROPERTY_TASK = {
    "allergen":         "classification",
    "halflife_invitro": "regression",
    "halflife_invivo":  "regression",
    "hemolysis":        "classification",
    "nonfouling":       "classification",
    "permeability":     "classification",
    "solubility":       "classification",
    "stability":        "regression",
    "toxicity":         "classification",
}

N_FOLDS = 5


# ─────────────────────────────────────────────────────────────────────
# Device helper
# ─────────────────────────────────────────────────────────────────────

def _resolve_device(gpu):
    """Parse gpu spec into a torch.device."""
    if gpu is None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    s = str(gpu)
    if s.isdigit():
        return torch.device(f"cuda:{s}")
    return torch.device(s)


# ─────────────────────────────────────────────────────────────────────
# Feature extraction (self-contained, no external aggregator needed)
# ─────────────────────────────────────────────────────────────────────

class FeatureExtractor:
    """Extract ESM2 embeddings, iupred2a IDS, and RDKit SMILES for a peptide."""

    def __init__(self, esm_device=None, iupred_dir=None):
        self.iupred_dir = iupred_dir or IUPRED_DIR
        self.esm_device = _resolve_device(esm_device)
        self._esm_tokenizer = None
        self._esm_model = None
        self._rdkit_ready = False
        self._normalizer = self._lfc = self._uc = self._Chem = None

    def _ensure_rdkit(self):
        if self._rdkit_ready:
            return
        from rdkit import Chem, RDLogger
        from rdkit.Chem import MolStandardize
        RDLogger.DisableLog("rdApp.*")
        self._normalizer = MolStandardize.normalize.Normalizer()
        self._lfc = MolStandardize.fragment.LargestFragmentChooser()
        self._uc = MolStandardize.charge.Uncharger()
        self._Chem = Chem
        self._rdkit_ready = True

    def _ensure_esm(self):
        if self._esm_model is not None:
            return
        from transformers import AutoModel, AutoTokenizer
        print(f"[predict] Loading ESM2 from {ESM_MODEL_PATH} on {self.esm_device}",
              file=sys.stderr)
        self._esm_tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_PATH)
        self._esm_model = AutoModel.from_pretrained(ESM_MODEL_PATH).to(self.esm_device)
        self._esm_model.eval()

    def seq_to_smiles(self, seq: str) -> str:
        """Convert peptide sequence to canonical SMILES via RDKit."""
        self._ensure_rdkit()
        seq_clean = seq.replace("X", "")
        if not seq_clean:
            raise ValueError(f"Empty sequence after stripping 'X': {seq!r}")
        mol = self._Chem.MolFromFASTA(seq_clean)
        if mol is None:
            raise ValueError(f"RDKit MolFromFASTA failed: {seq!r}")
        raw = self._Chem.MolToSmiles(mol)
        mol = self._Chem.MolFromSmiles(raw)
        mol = self._normalizer.normalize(mol)
        mol = self._lfc.choose(mol)
        mol = self._uc.uncharge(mol)
        return self._Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)

    def seq_to_ids(self, seq: str) -> np.ndarray:
        """Compute iupred2a disorder scores → (L, 3) array [long, short, anchor]."""
        iupred_script = os.path.join(self.iupred_dir, "iupred2a.py")
        if not os.path.exists(iupred_script):
            raise FileNotFoundError(f"iupred2a.py not found at {iupred_script}")
        with tempfile.TemporaryDirectory() as tmp:
            fasta_path = os.path.join(tmp, "pep.fasta")
            with open(fasta_path, "w") as f:
                f.write(f">peptide\n{seq}\n")

            def _run(*args):
                return subprocess.run(
                    ["python", iupred_script, *args],
                    capture_output=True, text=True, check=False,
                ).stdout

            long_out  = _run(fasta_path, "long")
            short_out = _run(fasta_path, "short")
            anchor_out = _run("-a", fasta_path, "long")

        def _parse(output, col_idx):
            scores = []
            for line in output.split("\n"):
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split()
                if len(parts) > col_idx:
                    try:
                        scores.append(float(parts[col_idx]))
                    except ValueError:
                        pass
            return scores

        long_s   = _parse(long_out, 2)
        short_s  = _parse(short_out, 2)
        anchor_s = _parse(anchor_out, 3)
        L = len(seq)
        if not (len(long_s) == len(short_s) == len(anchor_s) == L):
            raise RuntimeError(
                f"iupred2a length mismatch (L={L}, long={len(long_s)}, "
                f"short={len(short_s)}, anchor={len(anchor_s)})")
        return np.column_stack([long_s, short_s, anchor_s])

    def seq_to_esm(self, seq: str) -> torch.Tensor:
        """Extract ESM2 per-residue embeddings → (L, 1280) tensor on CPU."""
        self._ensure_esm()
        encoded = self._esm_tokenizer(
            seq, return_tensors="pt", padding=False,
            truncation=False, max_length=5120,
        )
        encoded = {k: v.to(self.esm_device) for k, v in encoded.items()}
        with torch.no_grad():
            out = self._esm_model(**encoded)
            emb = out.last_hidden_state.squeeze(0)[1:-1].cpu()  # strip CLS/EOS
        return emb


# ─────────────────────────────────────────────────────────────────────
# Property ensemble: 5-fold PTNet checkpoints → one sigmoid score
# ─────────────────────────────────────────────────────────────────────

class PropertyEnsemble:
    """Loads 5-fold PTNet checkpoints per property, predicts via mean-logit."""

    def __init__(self,
                 properties: List[str],
                 split: str = "random",
                 ckpt_root: str = None,
                 device=None,
                 fold_aggregation: str = "mean_logit"):
        self.properties = list(properties)
        self.split = split
        self.device = device or _resolve_device(None)
        self.fold_aggregation = fold_aggregation
        # ckpt_root: e.g. "/.../finetune_property2/output"
        self.ckpt_root = Path(ckpt_root) if ckpt_root else None
        self._models: Dict[str, List[torch.nn.Module]] = {}

    def _discover_ckpt_root(self, prop: str) -> Path:
        """Find the checkpoint root directory for a property."""
        if self.ckpt_root is not None:
            return self.ckpt_root
        # Priority: PEPD_CKPT_ROOT from config → PROJECT_DIR/output (legacy).
        from config import PEPD_CKPT_ROOT, PROJECT_DIR
        candidates = [
            Path(PEPD_CKPT_ROOT),
            Path(PROJECT_DIR) / "output",
        ]
        for c in candidates:
            if (c / prop).exists():
                return c
        raise FileNotFoundError(
            f"No checkpoint root found for {prop!r}. Tried: {candidates}")

    def _load(self, prop: str) -> List[torch.nn.Module]:
        if prop in self._models:
            return self._models[prop]

        root = self._discover_ckpt_root(prop)
        # Search order: <root>/<prop>/  (flat layout) → <root>/<prop>/<split>/  (legacy)
        search_dirs = [root / prop, root / prop / self.split]
        models = []
        for d in search_dirs:
            for fold in range(1, N_FOLDS + 1):
                p = d / f"fold_{fold}.pt"
                if not p.exists():
                    continue
                amnet = AMNet()
                smnet = SMNet()
                m = PTNet(amnet, smnet, mode="full").to(self.device)
                m.load_state_dict(torch.load(p, map_location=self.device))
                m.eval()
                for prm in m.parameters():
                    prm.requires_grad = False
                models.append(m)
            if models:
                break

        if not models:
            searched = " or ".join(str(d) for d in search_dirs)
            raise FileNotFoundError(
                f"No fold_*.pt checkpoints for {prop!r} under {searched}")
        self._models[prop] = models
        print(f"  [predict] loaded {prop}: {len(models)} fold(s)", file=sys.stderr)
        return models

    def warmup(self):
        """Eagerly load all checkpoints (useful before batch scoring)."""
        for p in tqdm(self.properties, desc="Loading property checkpoints"):
            self._load(p)

    @torch.no_grad()
    def predict(self, xpep, xsmi) -> Dict[str, float]:
        """Run all property ensembles on prepared PyG graphs."""
        out = {}
        for prop in self.properties:
            try:
                models = self._load(prop)
            except FileNotFoundError as e:
                print(f"  WARN: skip {prop}: {e}", file=sys.stderr)
                continue

            if self.fold_aggregation == "mean_logit":
                logits = torch.stack(
                    [m(xpep, xsmi).view(-1) for m in models], dim=0)
                d = torch.sigmoid(logits.mean(dim=0)).item()
            elif self.fold_aggregation == "mean_prob":
                probs = torch.stack(
                    [torch.sigmoid(m(xpep, xsmi).view(-1)) for m in models], dim=0)
                d = probs.mean(dim=0).item()
            else:
                raise ValueError(
                    f"Unknown fold_aggregation: {self.fold_aggregation}")
            out[prop] = float(d)
        return out


# ─────────────────────────────────────────────────────────────────────
# Main scorer: feature extraction + ensemble prediction
# ─────────────────────────────────────────────────────────────────────

class PeptidePropertyScorer:
    """Single peptide → 9 property desirability scores."""

    def __init__(self,
                 split: str = "random",
                 fold_aggregation: str = "mean_logit",
                 ckpt_root: str = None,
                 gpu=None,
                 esm_device=None):
        self.split = split

        self.extractor = FeatureExtractor(esm_device=esm_device)
        self.infer_device = _resolve_device(gpu)

        self.ensemble = PropertyEnsemble(
            properties=PROPERTIES,
            split=split,
            ckpt_root=ckpt_root,
            device=self.infer_device,
            fold_aggregation=fold_aggregation,
        )

    def warmup(self):
        """Eagerly load ESM2 + all property checkpoints."""
        self.extractor._ensure_esm()
        self.ensemble.warmup()

    @torch.no_grad()
    def score(self, sequence: str, name: str = "peptide") -> Dict:
        from torch_geometric.data import Batch

        seq = sequence.strip().replace("X", "")
        if not seq:
            raise ValueError("Empty sequence after stripping 'X'")

        # 1) Extract features
        smiles = self.extractor.seq_to_smiles(seq)
        ids = self.extractor.seq_to_ids(seq)
        emb = self.extractor.seq_to_esm(seq)

        # 2) Build PyG graphs (featurize_peptide_graph expects file path)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tf:
            tmp_path = tf.name
            torch.save(emb, tf)
        try:
            pep_data = featurize_peptide_graph(
                {"seq": seq, "ids": ids, "embed": tmp_path}, name=name)
            smi_data = featurize_smiles_graph({"seq": smiles}, name=name)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        xpep = Batch.from_data_list([pep_data]).to(self.infer_device)
        xsmi = Batch.from_data_list([smi_data]).to(self.infer_device)

        # 3) Predict all properties
        d = self.ensemble.predict(xpep, xsmi)

        return {
            "name": name,
            "sequence": seq,
            "smiles": smiles,
            "d": d,
            "split": self.split,
        }

    def score_many(self,
                   items: Iterable[Tuple[str, str]],
                   show_progress: bool = True) -> pd.DataFrame:
        """Score (name, sequence) pairs → DataFrame with d_<prop> columns."""
        rows = []
        it = list(items)
        bar = tqdm(it, desc="Scoring peptides") if show_progress else it
        for name, seq in bar:
            try:
                r = self.score(seq, name=name)
                row = {
                    "name": r["name"],
                    "sequence": r["sequence"],
                    "length": len(r["sequence"]),
                }
                for p in PROPERTIES:
                    row[f"d_{p}"] = r["d"].get(p)
                rows.append(row)
            except Exception as e:
                rows.append({
                    "name": name, "sequence": seq,
                    "length": len(seq),
                    "error": repr(e),
                })
        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def _read_input(path: Path) -> List[Tuple[str, str]]:
    """Read TSV/CSV with 'name' and 'sequence' columns."""
    sep = "\t" if str(path).endswith((".tsv", ".txt")) else ","
    df = pd.read_csv(path, sep=sep, engine="python")
    if "sequence" not in df.columns:
        raise ValueError(f"{path}: requires a 'sequence' column")
    if "name" not in df.columns:
        df["name"] = [f"peptide_{i}" for i in range(len(df))]
    return list(zip(df["name"].astype(str), df["sequence"].astype(str)))


def main():
    ap = argparse.ArgumentParser(
        description="Predict 9 peptide property scores from sequence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--sequence", default=None,
                    help="Single peptide sequence (e.g. 'GSPLRYQWTIDYKGR').")
    ap.add_argument("--name", default="peptide",
                    help="Name for the peptide (only with --sequence).")
    ap.add_argument("--input", type=Path, default=None,
                    help="TSV/CSV with 'name, sequence' columns for batch scoring.")
    ap.add_argument("--output", type=Path, default=None,
                    help="Output TSV path (default: <input>.scored.tsv).")
    ap.add_argument("--split", default="random",
                    choices=["random", "cluster_id30", "cluster_id50"],
                    help="Which split's checkpoints to use (default: random).")
    ap.add_argument("--fold_aggregation", default="mean_logit",
                    choices=["mean_logit", "mean_prob"],
                    help="How to aggregate 5-fold predictions (default: mean_logit).")
    ap.add_argument("--ckpt_root", type=Path, default=None,
                    help="Root dir containing <prop>/fold_*.pt checkpoints. "
                         "Default: auto-detect from PEPD_CKPT_ROOT config.")
    ap.add_argument("--gpu", default=None,
                    help="GPU for model inference (default: cuda:0 or cpu).")
    ap.add_argument("--esm_device", default=None,
                    help="Device for ESM2 model (default: same as --gpu). "
                         "Use 'cpu' to save GPU memory.")
    ap.add_argument("--warmup", action="store_true",
                    help="Eagerly load all checkpoints at startup.")
    args = ap.parse_args()

    if args.sequence is None and args.input is None:
        ap.error("Pass --sequence or --input")

    t0 = time.time()
    scorer = PeptidePropertyScorer(
        split=args.split,
        fold_aggregation=args.fold_aggregation,
        ckpt_root=str(args.ckpt_root) if args.ckpt_root else None,
        gpu=args.gpu,
        esm_device=args.esm_device,
    )
    if args.warmup:
        scorer.warmup()
    print(f"[predict] init done in {time.time() - t0:.1f}s", file=sys.stderr)

    # ── Single peptide ─────────────────────────────────────────────
    if args.sequence:
        r = scorer.score(args.sequence, name=args.name)
        out = {
            "name": r["name"],
            "sequence": r["sequence"],
            "smiles": r["smiles"],
            "d": {k: round(v, 4) for k, v in sorted(r["d"].items())},
        }
        print(json.dumps(out, indent=2))

        # Summary table
        print(f"\n{'Property':<20} {'d':>8}  {'Task':<16}  {'Description'}")
        print("-" * 72)
        desc = {
            "allergen":         "non-allergenic",
            "halflife_invitro": "long in-vitro half-life",
            "halflife_invivo":  "long in-vivo half-life",
            "hemolysis":        "non-hemolytic",
            "nonfouling":       "non-fouling",
            "permeability":     "high permeability",
            "solubility":       "high solubility",
            "stability":        "high stability",
            "toxicity":         "non-toxic",
        }
        for p in PROPERTIES:
            d_val = r["d"].get(p)
            d_str = f"{d_val:.4f}" if d_val is not None else "N/A"
            print(f"  {p:<18} {d_str:>8}  {PROPERTY_TASK[p]:<16}  {desc[p]}")

    # ── Batch ───────────────────────────────────────────────────────
    if args.input:
        items = _read_input(args.input)
        df = scorer.score_many(items)
        out_path = args.output or args.input.with_suffix(".scored.tsv")
        df.to_csv(out_path, sep="\t", index=False, float_format="%.4f")
        n_ok = df["error"].isna().sum() if "error" in df.columns else len(df)
        print(f"\n[predict] {n_ok}/{len(df)} scored, wrote → {out_path}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
