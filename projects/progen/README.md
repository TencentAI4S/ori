# Protein Generation

## Common Prompts

```python
prompts = [
    "<Glucosaminidase><temperature90><:>",
    "<Phage_lysozyme><temperature90><:>",
    "<Transglycosylas><temperature90><:>",
    "<Pesticin><temperature90><:>",
    "<Glyco_hydro_108><temperature90><:>",
    "<EC:3.1.1.101><temperature90><:>",
    "<EC:3.2.1.14><temperature90><:>",
    "<EC:3.2.1.17><temperature90><:>"
]
```

## Conditional generation 

```python
import torch
from tgnn.deploy import ProteinGenerator

progen = ProteinGenerator()
progen.load_state("pgm-3b")
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
progen.to(dtype)
prompt = "<Glucosaminidase><temperature90><:>"
progen.generate([prompt, ], num_samples=1)
```