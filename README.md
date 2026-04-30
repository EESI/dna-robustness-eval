# dna-robustness-eval

A framework for evaluating the robustness of DNA language models under sequence perturbations with biologically inspired constraints.

This repository provides perturbation operators and evaluation metrics for analyzing how model predictions change under controlled sequence variations while tracking key sequence-level properties.

---

## Features

### Perturbations
- Nucleotide substitution
- Codon substitution
- Synonymous codon substitution (amino acid preserving)
- GC-guided synonymous substitution (composition-aware)
- Backtranslation (protein-preserving re-encoding)

### Evaluation
- Attack Success Rate (ASR)
- GC-content deviation
- RNA folding stability (MFE via RNAfold)

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/dna-robustness-eval.git
cd dna-robustness-eval
pip install -r requirements.txt
````

Dependencies:

```
numpy
pandas
tqdm
```

RNAfold (ViennaRNA) is required for MFE computation.

---

## Usage

```python
import pandas as pd
from src.perturbations.gc_guided import synonymous_codon_attack_gc
from src.evaluation.gc_content import calculate_gc_content

sequences = pd.Series(["ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG"])

mutated = synonymous_codon_attack_gc(
    sequences,
    mutation_rate=0.2,
    lambda_gc=1.0,
    iteration=1
)

delta_gc = abs(
    calculate_gc_content(sequences.iloc[0]) -
    calculate_gc_content(mutated.iloc[0])
)
```

---

## Reproducibility

The repository provides core perturbation and evaluation components used in the paper.

To reproduce experiments:

1. Prepare a DNA sequence classification dataset
2. Run model inference on original sequences
3. Apply perturbations
4. Run inference on perturbed sequences
5. Compute robustness metrics

---

## Notes

* Perturbations are designed for robustness evaluation, not biological validity.
* GC content and RNA folding stability are proxy measures of sequence-level properties and do not fully capture biological function.
* Datasets and model checkpoints are not included.

---





## train dna language models usage

### Basic

```bash
python scripts/run_experiment.py \
  --train_csv path/to/train.csv \
  --valid_csv path/to/valid.csv \
  --test_csv path/to/test.csv \
  --model grover
````

### Models

```bash
--model {grover | dnabert2 | nt}
```

### Optional

```bash
--aug_csv path/to/aug.csv
--output_dir path/to/output_dir
```

### Example

```bash
python scripts/run_experiment.py \
  --train_csv data/train.csv \
  --valid_csv data/valid.csv \
  --test_csv data/test.csv \
  --model dnabert2 \
  --aug_csv data/aug.csv \
  --output_dir results/run1
```

### Output

* test_results.txt
* test_predictions.csv
* test_logits.npy
* confusion_matrix.png


