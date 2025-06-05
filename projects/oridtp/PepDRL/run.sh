#!/bin/bash

PLAYOUT_NUM=10
BATCH_SIZE=1
JUMPOUT=32
C_PUCT=0.5
NITER=300
WORKDIR='./'
FIRST_PDBID='7lll_R'
SEQ='HSQGTFTSDYSKYLDSQGRDFVQWLWLAGG'

SECOND_PDBID='5yqz_R'

OUTPUT_DIR="./results/${FIRST_PDBID}+${SECOND_PDBID}/${SEQ}"

python "$WORKDIR/train.py" \
    --pdbid "$FIRST_PDBID" \
    --dual_pdbid "$SECOND_PDBID" \
    --start_sequence "$SEQ" \
    --output_dir "$OUTPUT_DIR" \
    --n_playout "$PLAYOUT_NUM" \
    --batch_size "$BATCH_SIZE" \
    --jumpout "$JUMPOUT" \
    --c_puct "$C_PUCT" \
    --niter "$NITER"

echo "-------Finished!-------"