# Second checkpoint

### 5. Time-dependent explanations of neural networks for survival analysis

## Team
Jakub Skrajny, Maciej Domaradzki, Maurycy Moczulski

## Project goal
Compare SurvSHAP(t) model-agnostic explanation for survival models to explanations specific to neural networks, e.g. DeepLift. Hopefully model-specific explanations are comparable to SurvSHAP(t), but a lot faster to compute.

## What was done
- Train DeepHit on medical datasets:
  1. Heart Failure from SurvSHAP(t) paper
  2. Metabric from pycox
- Explain DeepHit using SurvSHAP(t)
- Explain DeepHit using DeepLift

## What will be done
- Apply mentioned methods to more datasets
- Compare SurvShap with DeepLift using Avg-Sensitivity, Faithfulness Correlation metrics
- Straighten notebooks and write down results