# VisMoE: A Knowledge-Decoupled Visiting MoE Framework for Cross-Species mRNA Translation Efficiency Prediction

This repository implements a **Mixture-of-Experts (MoE)** framework designed for predicting RNA Translation Efficiency (TE). It features a novel **Visiting Expert Strategy** to handle cross-species knowledge transfer (e.g., from Mouse to Human) and addresses domain shift in biological sequences.



##  Key Features

* **Multi-Modal Integration**: Combines raw codon sequences (1D-CNN), BERT-based latent embeddings, and hand-crafted biological features (GC content, codon frequency).
* **SwiGLU Experts**: Utilizes Gated Linear Unit experts with SiLU activation for enhanced non-linear representation.
* **Visiting Expert Strategy**: A transfer learning approach that identifies and fine-tunes the most "active" experts for the target species, preserving general biological patterns while adapting to specific ones.
* **Advanced Loss Functions**: Implements a hybrid loss comprising `SmoothL1Loss` for stability and `Pairwise Ranking Loss` to optimize the relative ordering of TE values.

---

##  Project Structure

| File | Description |
| :--- | :--- |
| `data_utils.py` | Data pipeline: Codon tokenization, bio-feature extraction, and `HybridDataset` class. |
| `model_moe.py` | Architecture: Sequence encoders, Top-K router, and the `AttnFusionMoE` main model. |
| `main.py` | Execution: Multi-phase training pipeline, evaluation metrics, and logging setup. |

---

##  Installation & Usage

### 1. Requirements
```bash
pip install torch pandas numpy scikit-learn scipy tqdm openpyxl
```

### 2. Running 
```bash
python main.py
```

##  Training Strategy (Three Phases)
Phase 1: Source Pre-training: The model is trained on the source species (Mouse) to learn generalizable RNA features.

Phase 2: Visiting Expert Adaptation: Based on usage frequency in the target domain, the top $K$ experts are identified as "Visiting Experts." Only these experts and the projection heads are unfrozen for target-domain (Human) adaptation.

Phase 3: Global Calibration: All weights (except the frozen low-level sequence encoders) are fine-tuned using a combination of MSE and Pairwise Ranking Loss to maximize correlation metrics (PCC/SCC).


# README
