# Exercise-Related Confounding in Wrist-Based Classification of Protocol-Defined Stress

Analysis code and derived window-level features for the article

> Huang, S.-F., & Huang, Y.-P. (in preparation). *Exercise-Related Confounding in Wrist-Based Classification of Protocol-Defined Stress: A Leave-One-Subject-Out Evaluation on a Public Dataset.*

The study uses only a **public dataset** (Hongn et al., 2025; PhysioNet) and evaluates whether wrist heart rate, pulse-rate variability (PRV) and acceleration discriminate protocol-defined stress in unseen participants, and whether such classifiers flag exercise as stress.

## Data source (not redistributed here)

- Hongn, A., Bosch, F., Prado, L., & Bonomini, P. (2025). *Wearable device dataset from induced stress and structured exercise sessions* (version 1.0.1). PhysioNet. https://doi.org/10.13026/he0v-tf17 (Open Data Commons Attribution License v1.0)
- Data descriptor: Hongn, A., Bosch, F., Prado, L., Ferrández, J., & Bonomini, M. (2025). *Wearable physiological signals under acute stress and exercise conditions.* Scientific Data, 12(1), 520. https://doi.org/10.1038/s41597-025-04845-9

Download the dataset from PhysioNet and unzip it. The scripts expect a folder that contains `wearable-device-dataset-…-1.0.1/Wearable_Dataset/{STRESS,AEROBIC,ANAEROBIC}` plus `Stress_Level_v1.csv` and `Stress_Level_v2.csv`.

## What is in this repository

| Folder | Content |
|---|---|
| `code/` | Analysis scripts, numbered in running order |
| `results/` | Derived window-level features (`windows.csv`, `seq.npy`), per-participant outputs, result tables (`*.csv`, `*.json`) and run logs (`*.log`) |

`results/windows.csv` contains 30-s window features derived from the public recordings (stress 359, rest 1,205, exercise 2,462 windows). It contains no raw signals and no personal data. Please cite the original dataset when reusing it (CC-BY-style attribution as required by the dataset licence).

## Requirements

Python 3.13 with `numpy 2.3`, `pandas 2.3`, `scipy 1.16`, `scikit-learn 1.7`, `torch 2.11` (CPU is sufficient), `statsmodels 0.14`, `matplotlib 3.10` (see `requirements.txt`).

## Reproducing the analyses

Run from the repository root.

```bash
# 1. Build 30-s windows and features (writes results/windows.csv, seq.npy, inventory.csv)
python code/01_build_windows.py DATA results

# 2. Leave-one-subject-out and cross-stage baselines (logistic regression, random forest, LSTM)
python code/02_loso_and_cross_stage.py results

# 3. Confidence intervals and feature ablations
python code/03_ci_and_ablation.py results

# 4. Leave-one-task-out, per-participant results, self-rated stress, selective classification
python code/04_extended_analyses.py results/windows.csv DS/Stress_Level_v1.csv DS/Stress_Level_v2.csv results

# 5. Exercise false alarms (rest-only training), paired model comparison, mixed models, random 10-fold
python code/05_priority_analyses.py results/windows.csv results/loso_probs.csv results/phase_selfreport.csv results

# 6. Exercise-inclusive training and distribution shift
python code/06_exercise_inclusive_and_shift.py results/windows.csv results

# 7. Figures
python code/07_make_figures.py results figures

# 8. LSTM vs logistic regression (paired), three-class models, out-of-range shares, Holm correction
python code/09_additional_checks.py results/windows.csv results/loso_probs.csv results

# 9. PRV-missing-flag and PRV-feature sensitivity
python code/10_prv_flag_sensitivity.py results/windows.csv results
```

## Notes

- **Participants.** 36 participants completed the stress protocol; S02 (duplicated signal values), f07 (protective dock not removed) and f14 (recording split across files) were excluded, leaving 33. Exercise recordings split across files (S11 aerobic, S16 anaerobic) were excluded. 29 participants have stress, rest and exercise data.
- **Labels.** "Stress" means protocol-defined stress (the task designated as stress-inducing in the original protocol), not individually experienced stress. Segment boundaries follow the button-press tags and the indexing used in the dataset's example notebook.
- **Randomness.** Random forests use a fixed seed. LSTM results use seeds 0–2; small numerical differences can occur across hardware and library versions.
- **Statistics.** Windows are model inputs, not independent units. Confidence intervals resample participants; paired comparisons use per-participant differences.

## Licence

Code licence: *to be chosen by the authors before publication* (for example MIT). Derived features follow the attribution requirements of the source dataset.

## Archived material

`archive_2026-10_simulated_dataset/` holds a synthetic dataset released earlier for a different manuscript. It is **not** used in this study (see the note in that folder).
