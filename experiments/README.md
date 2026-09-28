# Experiments

## nn_probe.py: can race be read from a model that never sees it?

### Question

The denial model's inputs come from an allowlist (`hmda.model.features.FEATURE_COLUMNS`). Race, ethnicity and sex are never inputs. So can race still be recovered from those inputs, or from the hidden layers of a small neural net trained on them? If it can, then dropping the race column does not make a model race-blind. The information still gets in through other fields.

A hidden layer is computed from the inputs, so it can never hold more race information than the inputs do. The probes measure how easy that information is to read out, not how much of it exists.

### Method

- Data: a stratified 500,000-row sample of the national HMDA file, one draw per seed. Rows are kept only if the action is originated, approved but not accepted, or denied. Train on 2023-2024 and test on 2025.
- Of the 2023-2024 rows, 10% are held out at random for validation. The imputer, scaler, network and both baselines are all fitted on the other 90%.
- Network: inputs, then 64 ReLU units, then 32 ReLU units, then one output. Adam at lr 1e-3. It trains for at most 30 epochs, stops early if validation loss has not improved for 4 epochs, and keeps the best epoch's weights.
- Baselines: the repo's logistic regression and LightGBM, fitted on the same rows.
- Probes: the Black and White applicants in the test set are split in half. Each probe is fitted on one half and scored by AUC on the other half. There are two kinds of probe:
  - linear: logistic regression on standardized features
  - MLP: one hidden layer of 64 units, stopped early on the log loss of a 10% slice of the fitting half. sklearn's built-in early stopping watches accuracy. When only about 11% of rows are positive, accuracy hardly moves, and in a smoke test that probe stopped close to a majority-class guess.
- Five seeds (0 to 4). Each seed changes the sample draw, the validation split, the weight init and the probe split. Std is the sample std (ddof=1).

### Controls

- Random-init layer 2: the same architecture with untrained weights. It shows what a probe gets from random ReLU features of the inputs.
- Shuffled labels: the same probe on hidden layer 2 with the race labels permuted. It should score about 0.5.
- The network's denial score used on its own as a race score.

### Results (SAMPLE numbers, mean ± std over 5 seeds)

Denial AUC on 2025:

| model | AUC |
|---|---|
| neural net | 0.838 ± 0.008 |
| logistic regression | 0.784 ± 0.001 |
| LightGBM | 0.859 ± 0.001 |

Race probe AUC (Black vs White, held-out half of the 2025 test rows):

| representation | linear probe | MLP probe |
|---|---|---|
| input features | 0.678 ± 0.004 | 0.688 ± 0.005 |
| hidden layer 1 | 0.673 ± 0.005 | 0.675 ± 0.005 |
| hidden layer 2 | 0.659 ± 0.006 | 0.672 ± 0.007 |
| random-init layer 2 (control) | 0.642 ± 0.007 | 0.664 ± 0.004 |
| shuffled labels, layer 2 (control) | 0.499 ± 0.003 | 0.502 ± 0.004 |

The network's denial score alone separates the two groups with AUC 0.580 ± 0.004.

Paired by seed, which means the same sample and the same probe split:

- An MLP probe on the inputs beats a linear probe by 0.009 ± 0.003, and it wins in all 5 seeds. So some of the race signal in the inputs is nonlinear, but not much.
- Linear probe, hidden layer 1 minus hidden layer 2: 0.014 ± 0.003, positive in all 5 seeds.
- Linear probe, inputs minus hidden layer 1: 0.005 ± 0.006. This is not clearly different from zero.
- Linear probe, trained layer 2 minus random-init layer 2: 0.017 ± 0.009, positive in all 5 seeds. With the MLP probe the gap is 0.008 ± 0.006, and one seed is slightly negative.

Early stopping fired in only 1 of 5 seeds (seed 1, best epoch 8). The other seeds reached their best epoch at 29 or 30, so they were still improving when they hit the cap. Longer training might move the network's AUC.

### Which inputs carry the race signal

This is a one-feature AUC for each input, written as max(auc, 1 - auc) so that direction does not matter. It is on the probe's scoring half, averaged over seeds:

| feature | one-feature AUC |
|---|---|
| property value | 0.599 ± 0.002 |
| income | 0.595 ± 0.004 |
| debt-to-income band | 0.594 ± 0.004 |
| loan-to-value ratio | 0.586 ± 0.003 |
| loan type: conventional | 0.584 ± 0.004 |

By the absolute standardized coefficient in the linear input probe, the features in the top 5 most often are property value (5 of 5 seeds), debt-to-income band (5 of 5) and conventional loan type (4 of 5). Positive means higher odds that the applicant is Black. Debt-to-income has a positive coefficient, and conventional loan type and property value have negative ones. The property value coefficient is unstable across seeds (-1.45 ± 1.00), probably because the raw value is heavy-tailed and overlaps with income and loan amount. Read the one-feature AUC as the more stable number. Code labels come from the CFPB LAR data field reference (https://ffiec.cfpb.gov/documentation/publications/loan-level-datasets/lar-data-fields).

### What this supports

- Race can be read from the allowed inputs at an AUC of about 0.68. It is not hidden by leaving out the race column. The signal comes from ordinary credit fields: property value, income, debt-to-income, loan-to-value and loan type.
- Training does not make race easier to read than it is from the raw inputs. Linear readability falls a little at each layer. Trained layer 2 is a little more readable than random features of the same shape.
- Most of the signal is linear. A nonlinear probe adds about 0.01 AUC.

### What this does not support

- It does not show that the model discriminates, that any lender does, or that there is disparate treatment in the legal sense. A probe measures what information is present. It does not measure intent, and it does not measure whether that information changes a decision.
- The layer-by-layer gaps are 0.01 to 0.02 AUC. They come from one architecture, one sample size, five seeds and a training run that mostly hit its epoch cap. Do not read them as general facts about neural networks.
- Every number here comes from a sample, not the full national file.
- The run is on Apple MPS, which is not bit-for-bit deterministic, so a rerun can differ in the last digits. The first single-seed run (6 epochs, no validation split) is not reproduced by seed 0 here, because the training setup changed.

### Run it

```
uv pip install -e ".[nn]"
.venv/bin/python experiments/nn_probe.py --n 500000 --seeds 0 1 2 3 4
```

Torch is an optional extra and is not part of the main dependencies. The run takes about 8 minutes on an M-series Mac (468 s summed over the five seeds). The aggregate and the per-seed detail are written to `results/nn_probe.json`. Per-seed files also go to `out/`, which is gitignored.
