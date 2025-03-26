# Protein Stucture Prediction

## ESMFold

The optimized ESM and ESMFold use Flash Attention and SDPA to accelerate Transformer and Evoformer.

```python
import torch
from tgnn.model.arch.esmfold import load_esmfold_model
from tgnn.protein.parser import parse_fasta

model = load_esmfold_model("esmfold_3B_v1").eval()
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
model.to(dtype)
if torch.cuda.is_available():
    model.cuda()

seqs, _, _ = parse_fasta("data/test.fasta")
aaseq = seqs[0]
if len(aaseq) > 1024:
    model.set_chunk_size(32)
else:
    model.set_chunk_size(None)
outputs = model.infer([aaseq, ], need_head=True)
```