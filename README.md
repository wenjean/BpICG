# BpICG

**Biology-guided multimodal learning for compound discovery and drug combination recommendation**

> **Repository status:** The code and data for this project are being organized and checked for release. This repository is currently a project overview; it does not yet contain a complete, runnable reproduction of the manuscript results. Training, inference, evaluation, and analysis code, together with data preparation instructions, will be added as they are finalized.

## Overview

BpICG is a framework for predicting how a compound changes gene expression in a given cellular context. It combines baseline gene expression, a pretrained Universal Cell Embedding (UCE), molecular structure represented by an FCFP4 fingerprint, drug concentration, and biological prior knowledge.

The biological priors include four gene-level graphs based on chromosomal proximity, Gene Ontology, KEGG, and transcription factor–target relationships. A separate KEGG pathway module uses drug-aware attention to model pathway-level responses. BpICG predicts post-treatment gene expression and supports downstream analyses of drug-induced transcriptional changes.

The study uses these predictions to:

1. Evaluate response prediction for compounds excluded from model training.
2. Interpret molecular features and pathway-level responses associated with predictions.
3. Rank single compounds by their predicted reversal of disease-associated gene-expression signatures.
4. Derive transcriptional and pathway-level features for drug-pair recommendation.

The manuscript presents applications to melanoma, hepatocellular carcinoma, and breast cancer models, including experimental evaluation of selected single compounds and drug combinations.

## Release preparation

We are consolidating the scripts used in the study into a documented workflow. The planned release includes:

- Data preprocessing, feature construction, and drug-disjoint train/validation/test splits.
- BpICG model definition, training, inference, and evaluation.
- Single-compound prioritization and drug-combination analysis.
- Scripts and configuration files for the principal figures and ablation experiments.
- Environment specifications, example commands, and descriptions of required input files.

Processed data, model checkpoints, and external resources are being reviewed for size and redistribution terms. Where files cannot be hosted in this repository, the documentation will provide their sources and preparation steps. **No installation or reproduction commands are provided yet because the release is still being assembled.**

## Data sources

The study analyzes L1000 perturbation profiles ([GEO: GSE92742](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742)), cancer transcriptomic data from the [NCI Genomic Data Commons](https://portal.gdc.cancer.gov/), and drug-combination response data from [DrugComb](https://drugcomb.org/). It also uses biological annotation resources and compound libraries described in the manuscript. Source-specific download links, versions, preprocessing details, and access requirements will be documented with the code release.


