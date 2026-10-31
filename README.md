# BpICG

Code for **Biology-Guided Multimodal Learning for Compound Discovery and Drug Combination Recommendation** by Jian Wen et al.

BpICG predicts drug-induced gene-expression responses using molecular fingerprints, baseline expression, pretrained cell embeddings, drug concentration, and biological priors. Predicted responses support compound prioritization through reversal of disease-associated expression signatures.

## Scripts

| Script | Purpose |
| --- | --- |
| [train_eval_drugs.py](model_training_evaluation/train_eval_drugs.py) | Train and evaluate BpICG at 10 µM using compound-disjoint training, validation, and test splits. |
| [train_eval_multidose.py](dose_response_evaluation/train_eval_multidose.py) | Train and evaluate responses across concentrations of `0 < dose <= 100` µM, with compounds held out across all their doses. |
| [recommend_drugs.py](drug_recommendation/recommend_drugs.py) | Predict candidate responses and rank single compounds by disease-signature reversal. |

The manuscript's drug-combination synergy analysis is a separate downstream workflow.

## Setup and data

Use **Python 3.10**. From the repository root, install the shared dependencies:

```bash
python -m pip install -r model_training_evaluation/requirements.txt
```

Processed data and pretrained model weights are available on Zenodo(https://doi.org/10.5281/zenodo.23589950) and Google Drive(https://drive.google.com/file/d/1HrHrnIEB-Jh12XyBT-E6WwVrT3rfYnsm/view?usp=drive_link). Place the required files at the relative paths specified in each workflow’s data_manifest.json, and keep each script together with its accompanying model and helper files.

## Usage

Run from the repository root after preparing the data:

```bash
python model_training_evaluation/train_eval_drugs.py 
python dose_response_evaluation/train_eval_multidose.py 
python drug_recommendation/recommend_drugs.py --cell_types A375 
```


Training saves checkpoints, data splits, and Pearson correlation, R², and MSE metrics. Recommendation saves predicted expression, log2 fold changes, and compound rankings. **Lower reversal scores indicate stronger predicted disease-signature reversal.**
