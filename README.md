# ProtDynPredict

**Predicting Protein Expression Dynamics from Intrinsic Features**

ProtDynPredict is a machine learning framework that predicts whether a protein will be differentially expressed in cancer using only protein-intrinsic properties — no prior expression data required. The model uses subcellular localization, GO annotations, PPI network topology, and mass spectrometry detectability features to classify proteins as upregulated, downregulated, or unchanged.

## Key Findings

- **Stage 1** (DE vs. unchanged): AUC = 0.709 (permutation p < 0.001)
- **Stage 2** (up vs. down): AUC = 0.709 (95% CI: 0.688–0.731)
- Outperforms all baselines including k-NN (AUC = 0.587) and label propagation (AUC = 0.569)
- GO-slim annotations (subcellular localization) carry the strongest predictive signal
- Cross-dataset generalization across 3 CPTAC cancer types (mean pairwise AUC = 0.647)
- Strong RNA-protein concordance in orthogonal validation (Spearman rho = 0.764)

## Model Architecture

A two-stage XGBoost classifier:

1. **Stage 1**: Differentially expressed vs. unchanged
2. **Stage 2**: Upregulated vs. downregulated (for Stage 1 positives)

Trained on 164 non-sequence features (GO-slim, PPI, detectability, pathway) after feature ablation showed that 999 sequence-derived descriptors introduce collinearity-induced noise.

## Features

| Category | Count | Description |
|----------|-------|-------------|
| GO-slim annotations | 150 | Binary membership across BP, MF, CC ontologies |
| PPI network | 3 | Degree, clustering coefficient, betweenness centrality |
| Detectability | 10 | Tryptic peptides, GRAVY, MW, pI, etc. |
| Pathway context | 1 | KEGG pathway membership count |

## Project Structure

```
ProtDynPredict/
├── R/                              # Feature engineering pipeline
│   ├── 00_run_pipeline.R           # Master pipeline script
│   ├── 01_fetch_protein_data.R     # UniProt metadata retrieval
│   ├── 02_sequence_features.R      # Amino acid descriptors (protr)
│   ├── 03_go_similarity.R          # GO semantic similarity
│   ├── 04_network_features.R       # STRING PPI network features
│   ├── 05_detectability_features.R # MS detectability proxies
│   ├── 06_assemble_features.R      # Merge all feature sets
│   └── 07_compute_predict_features.R
├── python/                         # ML pipeline
│   ├── 00_validate_premise.py      # Go/no-go masking experiment
│   ├── 01_train_model.py           # Two-stage XGBoost training
│   ├── 02_label_propagation.py     # Label propagation baseline
│   ├── 03_baselines.py             # Additional baselines (k-NN, random, majority)
│   ├── 04_validate_within.py       # Within-dataset validation
│   ├── 04b_feature_ablation.py     # Feature category ablation
│   ├── 05_validate_cross.py        # Cross-dataset generalization
│   ├── 06_orthogonal_validation.py # RNA-protein concordance
│   ├── 07_interpret.py             # SHAP interpretation
│   ├── 08_predict.py               # Prediction on undetected proteins
│   ├── 09_report.py                # Report generation
│   ├── 10_predict_undetected.py    # Predict undetected protein expression
│   ├── 11_annotation_bias_test.py  # Annotation bias analysis
│   ├── 12_supplementary_table_s4.py# Supplementary table generation
│   ├── 13_improved_model.py        # Non-sequence model with Optuna tuning
│   ├── fetch_cptac_data.py         # CPTAC data retrieval
│   ├── fetch_human_proteome.py     # Human proteome data fetching
│   └── utils/                      # Shared utilities
├── data/                           # Data directory
│   ├── raw/                        # User-uploaded DE results
│   ├── processed/                  # Feature matrices
│   └── external/                   # PaxDB, STRING, etc.
├── models/                         # Trained model files (.joblib)
├── results/                        # Output figures and reports
├── requirements.txt                # Python dependencies
└── PROJECT_PLAN.md                 # Detailed project plan
```

## Installation

### Python

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### R Dependencies

```r
install.packages(c("protr", "GOSemSim", "STRINGdb", "igraph",
                    "clusterProfiler", "ReactomePA", "biomaRt",
                    "UniProt.ws", "org.Hs.eg.db", "tidyverse"))
```

## Usage

### 1. Feature Engineering (R)

Run the R pipeline to compute protein features from UniProt IDs:

```bash
Rscript R/00_run_pipeline.R
```

This executes scripts 01–06 sequentially: fetching protein metadata, computing sequence descriptors, GO similarity, PPI network features, detectability proxies, and assembling the final feature matrix.

### 2. Validate Premise

```bash
python python/00_validate_premise.py
```

Masks 20% of detected proteins and tests whether intrinsic features predict their expression class. This is the go/no-go gate before full model training.

### 3. Train Model

```bash
python python/01_train_model.py
```

Trains the two-stage XGBoost classifier with Optuna hyperparameter optimization and grouped cross-validation.

### 4. Validate

```bash
python python/04_validate_within.py      # Within-dataset CV
python python/05_validate_cross.py       # Cross-dataset (UCEC/COAD/LUAD)
python python/06_orthogonal_validation.py # RNA-protein concordance
```

### 5. Interpret

```bash
python python/07_interpret.py            # SHAP analysis
python python/04b_feature_ablation.py    # Feature category ablation
```

### 6. Predict

```bash
python python/08_predict.py              # Predict on undetected proteins
python python/09_report.py               # Generate report
```

## Data

The model was trained and validated on CPTAC proteomics data:

| Dataset | Proteins | Samples (Tumor/Normal) |
|---------|----------|----------------------|
| UCEC    | 3,117    | 103 / 49             |
| COAD    | 2,326    | —                    |
| LUAD    | 3,107    | —                    |

Data is retrieved programmatically via the `cptac` Python package and UniProt REST API.

## Validation

- **Permutation test**: 1,000 label shuffles, empirical p < 0.001
- **Bootstrap CI**: 1,000 resamples for AUC, F1, MCC
- **Cross-dataset**: Pairwise and leave-one-cancer-out across 3 CPTAC cancer types
- **Orthogonal**: Matched transcriptomics (Spearman rho = 0.764, Fisher OR = 7.47)
- **MNAR simulation**: Performance stratified by detectability
- **Annotation bias test**: Controls for GO annotation depth confounding

## Citation

If you use ProtDynPredict in your research, please cite:

> Awoniyi, L. ProtDynPredict: Protein-Intrinsic Features Predict Tumor Expression Dynamics Across Cancer Types. (2025).

## License

This project is available for academic and research use.
