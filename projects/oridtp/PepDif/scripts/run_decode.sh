python -u run_decode.py \
--model_path ../model_weights/PepDif/PepDif.pt \
--seed 0 \
--split test \
--top_p -1 \
--step 2000 \
--bsz 60
