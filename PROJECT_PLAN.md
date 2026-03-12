# ProtDynPredict: Predicting Expression Dynamics of Undetected Proteins

## Problem Statement
Proteomics experiments routinely detect only 30-60% of the expressed proteome. Proteins below the detection limit, with poor ionization, or suppressed by high-abundance proteins are systematically missed. Current imputation methods (Perseus, DreamAI) only handle proteins missing in *some* samples — they cannot predict expression states for proteins **never detected in any sample**.

**Goal**: Build an ML model that predicts whether an undetected protein is upregulated, downregulated, or unchanged in a given experiment, using protein features and the expression context of detected proteins.

---

## Input Data
- Differential expression results: `UniProt_ID, log2FC, adj_pvalue`
- User provides their own DE proteomics output (CSV)

## Output
- For each undetected protein of interest: predicted class (up/down/unchanged) with confidence score
- Ranked prioritization list for experimental validation (PRM/MRM/Western blot)

---

## Phase 1: Data Collection & Feature Engineering (R)

### 1.1 Fetch Protein Metadata
**Script**: `R/01_fetch_protein_data.R`

```
Input: list of UniProt IDs (detected + candidate undetected proteins)
Process:
  - Use UniProt.ws or biomaRt to fetch:
    - Protein sequences (FASTA)
    - GO annotations (BP, MF, CC)
    - Subcellular localization
    - Molecular weight, pI
    - Pfam domains
  - Use STRINGdb to fetch PPI network (confidence > 400)
  - Use clusterProfiler/ReactomePA for pathway membership (KEGG + Reactome)
Output: protein_metadata.rds
```

### 1.2 Calculate Sequence Features
**Script**: `R/02_sequence_features.R`

```
Input: protein sequences from 1.1
Process (using protr):
  - Amino acid composition (AAC) — 20 features
  - Dipeptide composition (DC) — 400 features
  - CTD descriptors (composition, transition, distribution) — 147 features
  - Pseudo amino acid composition (PseAAC, lambda=10) — 30 features
  - Conjoint triad descriptors — 343 features
  - SKIP: Moreau-Broto autocorrelation (noisy for non-structure tasks)
Output: sequence_features.csv (~940 features per protein)
```

### 1.3 Calculate GO Semantic Similarity
**Script**: `R/03_go_similarity.R`

```
Input: GO annotations from 1.1, detected protein labels (up/down/unchanged)
Process (using GOSemSim):
  - Encode GO terms as GO-slim binary vector (~150 dimensions) — experiment-independent
  - For each protein, compute Wang semantic similarity to:
    - Top-5 most similar upregulated proteins (mean similarity)
    - Top-5 most similar downregulated proteins (mean similarity)
    - Top-5 most similar unchanged proteins (mean similarity)
  - Separate for BP, MF, CC ontologies → 9 summary features
  CAUTION: Group statistics must be recomputed per CV fold to avoid leakage
Output: go_features.csv (~159 features: 150 GO-slim + 9 summary)
```

### 1.4 Calculate Network Features
**Script**: `R/04_network_features.R`

```
Input: STRING PPI network, detected protein labels
Process (using igraph + STRINGdb):
  - For each protein, from its PPI neighbors that ARE detected:
    - Fraction of neighbors upregulated
    - Fraction of neighbors downregulated
    - Fraction of neighbors unchanged
    - Weighted versions (by STRING confidence score)
    - Number of detected neighbors (coverage)
  - Network topology:
    - Degree centrality
    - Betweenness centrality
    - Clustering coefficient
  - Pathway context:
    - For each pathway the protein belongs to:
      fraction of detected pathway members that are up/down/unchanged
    - Summarize as: max/mean pathway_fraction_up, pathway_fraction_down
  CAUTION: All neighbor/pathway expression stats must be recomputed per CV fold
Output: network_features.csv (~15 features)
```

### 1.5 Calculate Detectability Features
**Script**: `R/05_detectability_features.R`

```
Input: protein metadata from 1.1
Process:
  - Number of theoretical tryptic peptides (cleaveR or custom)
  - Molecular weight
  - Isoelectric point (pI)
  - GRAVY score (grand average of hydropathy)
  - Basal protein abundance (from PaxDB or Human Protein Atlas, if available)
  - Transmembrane domain count (from UniProt topology)
Output: detectability_features.csv (~6 features)
```

### 1.6 Assemble Feature Matrix
**Script**: `R/06_assemble_features.R`

```
Input: all feature CSVs from 1.2-1.5
Process:
  - Merge by UniProt ID
  - Flag detected vs. undetected proteins
  - Assign labels for detected proteins:
    - Upregulated:   log2FC > 1.0 AND adj_p < 0.05
    - Downregulated:  log2FC < -1.0 AND adj_p < 0.05
    - Unchanged:     adj_p > 0.20 AND |log2FC| < 0.5
    - EXCLUDE:       ambiguous zone (adj_p 0.05-0.20 or |log2FC| 0.5-1.0)
  - Export separate train (detected+labeled) and predict (undetected) sets
Output: feature_matrix_train.csv, feature_matrix_predict.csv
```

---

## Phase 2: Validate the Premise (Python)

**THIS IS THE GO/NO-GO GATE. Do this before building the full model.**

### 2.1 Masking Experiment
**Script**: `python/00_validate_premise.py`

```
Input: feature_matrix_train.csv (detected proteins only)
Process:
  - Randomly mask 20% of detected proteins (simulate "undetected")
  - Train XGBoost on remaining 80% → predict masked 20%
  - Use protein-family-grouped splits (CD-HIT at 40% identity)
  - Repeat 5 times with different random masks
  - Compute per-class AUC, macro-F1, MCC
Decision:
  - AUC > 0.65 for all classes → PROCEED to Phase 3
  - AUC 0.60-0.65 → Review feature importance, add features, retry
  - AUC < 0.60 → PIVOT (consider hybrid with transcriptomics, or reframe problem)
Output: validation_report.md with metrics and feature importance
```

---

## Phase 3: Model Training (Python)

### 3.1 Two-Stage XGBoost
**Script**: `python/01_train_model.py`

```
Architecture:
  Stage 1 — Binary: "differentially expressed" vs. "unchanged"
  Stage 2 — Binary: "upregulated" vs. "downregulated" (only for Stage 1 positives)

Rationale: The mechanism determining IF a protein changes differs from
WHAT DIRECTION it changes. Two focused models > one 3-class model.
Also mitigates class imbalance (unchanged is ~75% in 3-class).

Training:
  - XGBoost with scale_pos_weight for class imbalance
  - Hyperparameter tuning via Optuna (50 trials)
    - max_depth: [3, 8]
    - learning_rate: [0.01, 0.3]
    - n_estimators: [100, 1000]
    - subsample: [0.6, 1.0]
    - colsample_bytree: [0.3, 1.0]
    - min_child_weight: [1, 10]
  - 5-fold protein-family-grouped cross-validation
  - CRITICAL: Recompute network/GO group features per fold (no leakage)

Output: stage1_model.joblib, stage2_model.joblib, cv_results.csv
```

### 3.2 Label Propagation Baseline
**Script**: `python/02_label_propagation.py`

```
Baseline to beat:
  - Build PPI graph from STRING
  - Assign detected protein labels as initial states
  - Propagate labels through network using semi-supervised label propagation
  - Compare against XGBoost predictions
  If XGBoost cannot beat this, network features alone carry the signal
  and the sequence/GO features add no value.
Output: baseline_results.csv
```

### 3.3 Additional Baselines
**Script**: `python/03_baselines.py`

```
  1. Random prediction (respecting class priors)
  2. Majority class (everything = "unchanged")
  3. Nearest-neighbor by sequence similarity (BLAST top hit label)
  4. Label propagation (from 3.2)
All baselines evaluated with same grouped CV splits for fair comparison.
Output: all_baselines_comparison.csv
```

---

## Phase 4: Validation & Interpretation (Python)

### 4.1 Within-Experiment Validation
**Script**: `python/04_validate_within.py`

```
  - Leave-one-out: for each detected protein, pretend missing, predict class
  - Protein-family-grouped 5-fold CV (primary metric)
  - Metrics: macro-AUC, per-class precision/recall/F1, MCC, calibration plot
  - Confusion matrix analysis
Output: within_experiment_validation.csv, figures/
```

### 4.2 Cross-Experiment Validation (if multiple datasets available)
**Script**: `python/05_validate_cross.py`

```
  - Train on experiment A, test on experiment B
  - Tests whether model learns biology vs. memorizes experiment-specific patterns
  - Expected: lower performance, but informative
Output: cross_experiment_validation.csv
```

### 4.3 Orthogonal Validation
**Script**: `python/06_orthogonal_validation.py`

```
  - For predicted upregulated proteins, check:
    - Is the mRNA upregulated in matched transcriptomics? (if available)
    - Is it reported in literature for this condition?
    - Does it belong to pathways known to be activated?
  - Enrichment analysis of predicted DE proteins vs random
Output: orthogonal_validation_report.md
```

### 4.4 Model Interpretation
**Script**: `python/07_interpret.py`

```
  - SHAP values for both stage-1 and stage-2 models
  - Feature importance ranking
  - Key question to answer: Do network features dominate?
    If yes → the model is essentially doing sophisticated label propagation
    If sequence/GO features contribute meaningfully → novel biological insight
Output: shap_plots/, feature_importance.csv
```

---

## Phase 5: Prediction Pipeline & Output

### 5.1 Predict Undetected Proteins
**Script**: `python/08_predict.py`

```
Input: feature_matrix_predict.csv (undetected proteins)
Process:
  - Stage 1: predict DE probability
  - Stage 2: predict direction (up/down) for high-confidence DE
  - Calibrate probabilities (Platt scaling from CV)
  - Rank by confidence
Output: predictions.csv with columns:
  UniProt_ID, predicted_class, confidence, DE_probability,
  direction_probability, n_detected_neighbors, top_pathway_context
```

### 5.2 Generate Report
**Script**: `python/09_report.py`

```
Output: prediction_report.md containing:
  - Summary statistics
  - Top 50 highest-confidence predictions with biological context
  - Pathway enrichment of predicted DE proteins
  - Recommended validation targets (top 20)
  - Model performance summary and caveats
```

---

## Directory Structure

```
ML_project/
├── PROJECT_PLAN.md              ← this file
├── CLAUDE.md                    ← project instructions for Claude
├── R/
│   ├── 01_fetch_protein_data.R
│   ├── 02_sequence_features.R
│   ├── 03_go_similarity.R
│   ├── 04_network_features.R
│   ├── 05_detectability_features.R
│   └── 06_assemble_features.R
├── python/
│   ├── 00_validate_premise.py   ← RUN THIS FIRST
│   ├── 01_train_model.py
│   ├── 02_label_propagation.py
│   ├── 03_baselines.py
│   ├── 04_validate_within.py
│   ├── 05_validate_cross.py
│   ├── 06_orthogonal_validation.py
│   ├── 07_interpret.py
│   ├── 08_predict.py
│   └── 09_report.py
├── data/
│   ├── raw/                     ← user uploads DE results here
│   ├── processed/               ← feature matrices
│   └── external/                ← PaxDB, STRING, etc.
├── models/                      ← saved trained models
├── results/
│   ├── figures/
│   └── reports/
└── sample_data/                 ← example DE datasets for testing
```

---

## R Dependencies
```r
install.packages(c("protr", "GOSemSim", "STRINGdb", "igraph",
                    "clusterProfiler", "ReactomePA", "biomaRt",
                    "UniProt.ws", "org.Hs.eg.db", "tidyverse"))
```

## Python Dependencies
```bash
pip install xgboost scikit-learn optuna shap pandas numpy
pip install matplotlib seaborn joblib
```

---

## Key Risks & Mitigations

| Risk | Mitigation | Stage |
|------|-----------|-------|
| Core premise fails (AUC < 0.60) | Phase 2 go/no-go gate before investing in full pipeline | Phase 2 |
| MNAR domain shift (train on detectable, predict on undetectable) | Detectability features + down-weight high-abundance training proteins | Phase 3 |
| Data leakage from group features | Recompute network/GO group stats per CV fold | Phase 3 |
| Inflated metrics from protein families | CD-HIT clustered grouped cross-validation | Phase 2-4 |
| Class imbalance (75% unchanged) | Two-stage model + class weights + macro-averaged metrics | Phase 3 |
| Model just learns label propagation | Compare vs. label propagation baseline explicitly | Phase 3 |

---

## Success Criteria

- **Minimum viable**: Stage-1 AUC > 0.70, beats all baselines by > 0.05 AUC
- **Good**: Stage-1 AUC > 0.75, Stage-2 AUC > 0.80, cross-experiment AUC > 0.60
- **Publishable**: Above + orthogonal validation shows enrichment + SHAP reveals interpretable biology + tested on ≥3 independent datasets

---

## Execution Order

```
1. Set up directory structure and install dependencies
2. Obtain a test DE dataset (public proteomics from PRIDE/ProteomeXchange)
3. R scripts 01-06 (feature engineering) — sequential
4. Python 00 (VALIDATE PREMISE) — GO/NO-GO DECISION
5. Python 01-03 (model + baselines) — parallel where possible
6. Python 04-07 (validation + interpretation)
7. Python 08-09 (prediction + reporting)
```

## Estimated Dataset for Development
Use a public cancer proteomics dataset from PRIDE (e.g., PXD006109 or similar) with:
- Deep coverage (6,000+ proteins)
- Clear condition comparison (tumor vs. normal)
- Well-characterized biology (known pathways for validation)
