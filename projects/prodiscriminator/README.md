# Protein Property Prediction

## Solubility Prediction

```python
from tgnn.deploy import ProteinDiscriminator
from tgnn.protein.parser import parse_fasta
classifier = ProteinDiscriminator()
classifier.load_state(model="usm_100m_solubility")
seqs, _, _ = parse_fasta("data/solubility_demo.fasta")
classifier.predict([seqs, ], softmax=True)
```

## Thermostability Prediction

```python
from tgnn.deploy import ProteinDiscriminator
from tgnn.protein.parser import parse_fasta
classifier = ProteinDiscriminator()
classifier.load_state(model="usm_100m_thermostability")
seqs, _, _ = parse_fasta("data/thermostability_demo.fasta")
classifier.predict(seqs, softmax=True)
```


## SignalP Prediction

```python
from tgnn.deploy import ProteinDiscriminator
from tgnn.protein.parser import parse_fasta
classifier = ProteinDiscriminator()
classifier.load_state(model="usm_100m_signalp")
seqs, _, _ = parse_fasta("data/signalp_demo.fasta")
classifier.predict([{"seq": seqs[0], "kingdom": "EUKARYA"}, ], plot=True)
```