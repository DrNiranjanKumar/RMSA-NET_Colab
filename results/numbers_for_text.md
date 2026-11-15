# Numbers for the manuscript text (generated)

Every number below is computed from per_run.csv. Copy them into the text; do not type numbers by hand. p-values are two-sided paired t-tests over seeds (see stats.csv).


## Headline (abstract E-02, introduction E-08)

- **Clean** — MAE 0.926, Corr 0.671, Acc-7 35.42%, Acc-2 77.14%, F1 77.17%
- **A+V 30% + MMR** — MAE 0.955, Corr 0.664, Acc-7 34.99%, Acc-2 76.06%, F1 76.10%
- **A+V 30% zero-fill** — MAE 0.937, Acc-2 76.79%
- **MMR vs zero-fill (A+V 30%)** — ΔMAE = +0.018 (paired p = 0.375, 95% CI [-0.031, 0.067], better in 2/5 seeds)
- **MMR vs zero-fill (A+V 50%)** — ΔMAE = -0.018 (paired p = 0.238, 95% CI [-0.054, 0.018], better in 4/5 seeds)
- **Decomposition A+V30 vs clean** — total ΔMAE +0.029 = zero-fill part +0.011 + MMR part +0.018 (paired p = 0.202, 95% CI [-0.024, 0.081], better in 2/5 seeds)

## Training-time dropout vs test-time missingness (Table 7; E-41, E-44, E-60)

- **Clean test** — dropout-trained MMR 0.957 vs clean-trained 0.926 → Δ +0.031 (paired p = 0.179, 95% CI [-0.022, 0.085], better in 2/5 seeds)
- **A+V30 test, clean-trained model** — MMR 0.980 vs zero-fill 0.934 (paired p = 0.072, 95% CI [-0.007, 0.099], better in 0/5 seeds); MMR-rec trained clean: 0.934
- **Cost of missing input at test** — dropout-trained MMR: clean 0.957 → A+V30 0.955

## Ablation (Table 4; E-53, E-54, abstract PSG gain)

- **PSG (full vs CMTA+MMR)** — ΔAcc-7 +0.41 pp, ΔMAE +0.019 (paired p = 0.643, 95% CI [-1.858, 2.674], better in 2/5 seeds)
- **CMTA (full vs PSG+MMR)** — ΔAcc-7 -0.41 pp, ΔMAE +0.022 (paired p = 0.602, 95% CI [-2.413, 1.597], better in 1/5 seeds)
- **MMR (full vs PSG+CMTA zero-fill)** — ΔAcc-7 -0.90 pp, ΔMAE +0.018 (paired p = 0.285, 95% CI [-2.937, 1.130], better in 2/5 seeds)
- **All modules (full vs base)** — ΔAcc-7 -0.61 pp, ΔMAE +0.021 (paired p = 0.138, 95% CI [-1.530, 0.305], better in 1/5 seeds)

## MMR surrogate (Table 8; E-23, E-47)

- **MMR-task** — cos(f̂a,fa) 0.306, cos(f̂v,fv) 0.203, MAE 0.955
- **MMR-rec** — cos(f̂a,fa) 0.968, cos(f̂v,fv) 0.869, MAE 0.945
- **Zero-fill (untrained R)** — cos(f̂a,fa) -0.050, cos(f̂v,fv) -0.060, MAE 0.937
- **Text-MLP control** — cos(f̂a,fa) -0.001, cos(f̂v,fv) 0.032, MAE 0.915
- **MMR vs text-MLP** —  (paired p = 0.144, 95% CI [-0.021, 0.100], better in 2/5 seeds)
- **MMR-rec vs MMR-task** —  (paired p = 0.693, 95% CI [-0.070, 0.051], better in 3/5 seeds)

## Unaligned / CMTA (Tables 3, 10; E-43, E-48)

- **Unaligned vs aligned** — Acc-7 36.33 vs 35.42 (Δ +0.90); Acc-2 77.08 vs 77.14; MAE 0.914 vs 0.926 (paired p = 0.356, 95% CI [-1.502, 3.310], better in 2/5 seeds)
- **Unaligned: CMTA vs mean-pool** — ΔAcc-7 -0.12 (paired p = 0.909, 95% CI [-2.781, 2.548], better in 4/5 seeds)
- **Attention entropy (1 = uniform)** — audio 0.697, vision 0.528
- **Joint degradation (Table 6)** — MAE 0.941, Acc-7 36.18, Acc-2 76.85

## Baselines (Table 3b; E-39)

- **LFLSTM** — clean MAE 0.956; A+V30 MAE 0.957; A+V50 MAE 0.955
- **TFN** — clean MAE 1.003; A+V30 MAE 0.969; A+V50 MAE 0.975
- **MULT** — clean MAE 0.949; A+V30 MAE 0.953; A+V50 MAE 0.956
- **RMSA vs MulT (A+V30)** —  (paired p = 0.946, 95% CI [-0.048, 0.050], better in 3/5 seeds)
- **RMSA vs TFN (A+V30)** —  (paired p = 0.442, 95% CI [-0.062, 0.033], better in 3/5 seeds)
- **TFR-Net official, frame missing 0** — MAE 0.804 ± 0.022, Acc-7 41.14, Acc-2 80.99 (5 seeds)
- **TFR-Net official, frame missing 0.3** — MAE 0.770 ± 0.020, Acc-7 42.01, Acc-2 81.37 (5 seeds)
- **TFR-Net official, frame missing 0.5** — MAE 0.774 ± 0.009, Acc-7 42.24, Acc-2 81.28 (5 seeds)

## Robustness to partial / noisy input (Table 9; E-22, E-61)

- **frame30** — A+V30-trained MMR MAE 0.959 vs zero-fill 0.938 vs clean-trained 0.926
- **noise0.5** — A+V30-trained MMR MAE 0.958 vs zero-fill 0.941 vs clean-trained 0.926
- **noise1.0** — A+V30-trained MMR MAE 0.960 vs zero-fill 0.941 vs clean-trained 0.930

## MOSEI (Table 5)

- **MOSEI** — clean MAE —; A+V30 MMR —; zero-fill —

## Protocol facts (Section IV; E-31, E-32, E-34, E-51)

- **mosi constant predictor** — MAE 1.464 (predicting the training mean)
- **mosi data** — dims [768, 5, 20], max lengths T/Ta/Tv = 50/50/50, training segments with all-zero audio 0 / vision 1 of 1284
- **mosi_unal constant predictor** — MAE 1.464 (predicting the training mean)
- **mosi_unal data** — dims [768, 5, 20], max lengths T/Ta/Tv = 50/375/500, training segments with all-zero audio 0 / vision 1 of 1284
- **Parameters (full model)** — 1,983,078
- **Best epoch (A+V30 full)** — range 9–21 over 5 seeds
- **Compute** — 4.6 GPU-hours for 180 runs (36 configurations × 5 seeds)
