#!/bin/bash
# =============================================================================
# Run PepD (zenodo weights) prediction on 6 peptides via GPU server
#
# Usage on GPU server:
#   cd /owenbhe/buddy1/tylerlv/private/affinity/finetune_property3/ORI-PepCompass/PepD
#   bash run_predict_pepd.sh
# =============================================================================

export PATH="/root/miniconda3/bin:$PATH"
export PEPD_CKPT_ROOT="/owenbhe/buddy1/tylerlv/private/affinity/finetune_property3/PepD_zenodo/PepD"
export ESM_MODEL_PATH="/owenbhe/buddy1/tylerlv/private/Discriminator/Peptide_pretraining/output_clust/final/models--facebook--esm2_t33_650M_UR50D"

CODE_DIR="/owenbhe/buddy1/tylerlv/private/affinity/finetune_property3/ORI-PepCompass/PepD"
INPUT="/owenbhe/buddy1/tylerlv/private/affinity/finetune_property3/code/peptides_6.txt"
OUTPUT="/owenbhe/buddy1/tylerlv/private/affinity/finetune_property3/peptides_6_pepd_zenodo.tsv"

cd "$CODE_DIR"

echo "=== Environment ==="
echo "PEPD_CKPT_ROOT=$PEPD_CKPT_ROOT"
echo "ESM_MODEL_PATH=$ESM_MODEL_PATH"
python --version
python -c "import torch; print('CUDA:', torch.cuda.is_available(), 'devices:', torch.cuda.device_count()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo ""
echo "=== Predicting 6 peptides with PepD zenodo weights ==="
python predict.py \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --gpu 0 \
    --warmup

echo ""
echo "=== Results ==="
cat "$OUTPUT"
echo ""
echo "Done → $OUTPUT"
