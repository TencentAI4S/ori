#!/bin/bash

PLAYOUT_NUM=20
BATCH_SIZE=1
JUMPOUT=32
C_PUCT=0.5
NITER=100
WORKDIR='./'
PDBID='7lll_R'
SEQ='QDEEGLLLMQSLEMS'
OUTPUT_DIR="./results/${PDBID}/${SEQ}"

echo "Target PDB: $PDBID, Start sequence: $SEQ"

python "$WORKDIR/train.py" \
    --pdbid "$PDBID" \
    --start_sequence "$SEQ" \
    --output_dir "$OUTPUT_DIR" \
    --n_playout "$PLAYOUT_NUM" \
    --batch_size "$BATCH_SIZE" \
    --jumpout "$JUMPOUT" \
    --c_puct "$C_PUCT" \
    --niter "$NITER"

echo "-------Finished!-------"