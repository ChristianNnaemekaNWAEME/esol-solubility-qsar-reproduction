# esol-solubility-qsar-reproduction

Reproduces the classic ESOL aqueous-solubility QSAR model (Delaney, *J. Chem. Inf. Comput. Sci.* 2004, 44, 1000-1005) with RDKit, then tests the SLTS+LASSO estimator (Nwaeme, Lukman et al.) in the setting it was designed for: a continuous response with outliers.

## Data
`delaney-processed.csv` from the DeepChem repository (`deepchem/deepchem`, `datasets/`): 1,128 compounds with measured log solubility (mol/L). Download it into the same folder as `esol.py`.

## Part A: reproduction of the published model
Delaney's equation: logS = 0.16 - 0.63 cLogP - 0.0062 MW + 0.066 RB - 0.74 AP (RB: rotatable bonds, AP: aromatic atoms / heavy atoms). With RDKit descriptors (Crippen MolLogP in place of the original cLogP):

| | R2 | RMSE (log units) |
|---|---|---|
| Published equation, RDKit descriptors, all 1,128 compounds | 0.725 | 1.10 |
| Predictions stored in the DeepChem file | 0.811 | 0.91 |

The two sets of predictions correlate at 0.965. Refitting the four-term model on a scaffold split gives intercept 0.28 and coefficients logP -0.78, MW -0.0058, RB -0.014, AP -0.56, against the published 0.16, -0.63, -0.0062, +0.066, -0.74. The signs and sizes of logP, MW and AP agree; the rotatable-bond term does not (small and negative here, positive in the paper).
**Why this is a partial reproduction:** the paper fitted a different, larger compound set; logP comes from a different method; and the split here is by scaffold.

## Part B: robust estimation with gross outliers
About 730 descriptors (217 RDKit descriptors + 512-bin Morgan count fingerprint). A share of the *training* solubility values (0, 5, 10 or 20 percent) is replaced by gross outliers (shifted by 4 to 8 log units); test values are never altered. Five scaffold splits. Mean test RMSE (lower is better):

| model | 0% | 5% | 10% | 20% |
|---|---|---|---|---|
| Delaney 4-term (refit) | 1.105 | 1.112 | 1.145 | 1.136 |
| LASSO | 0.845 | 0.912 | 0.992 | 1.043 |
| Random Forest | 0.885 | 0.933 | 0.997 | 1.105 |
| SLTS+LASSO (keep 75%) | 0.877 | 0.882 | 0.875 | 0.880 |
| SLTS+LASSO (keep 90%) | 0.860 | 0.870 | 0.867 | 1.356 |

Reading: with clean data LASSO is slightly best. As outliers are added, LASSO and Random Forest degrade steadily, while SLTS+LASSO that keeps 75 percent of records stays flat. Keeping 90 percent works until contamination reaches 20 percent, which exceeds the 10 percent trimmed, and then it fails badly (RMSE 1.36, with large variance across splits). The trimming level must therefore exceed the contamination rate.

## Limitations
Synthetic outliers, one data set, five splits, untuned Random Forest. Solubility is an ADME property, not toxicity. The contamination here (gross outliers in a continuous response) is the setting of the published estimator, so these results do not by themselves show robustness for toxicity labels (see the toxicity-estimator-benchmark repository).

## Reproduce
```
pip install rdkit scikit-learn numpy pandas
python esol.py        # about 15 minutes on two CPU cores
```
