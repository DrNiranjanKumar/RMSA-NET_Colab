# RMSA-Net: Robust Multimodal Sentiment Analysis under Missing Modalities and Temporal Misalignment

Code, per-seed results and the scripts that generate every table and figure of the paper

> N. Kumar, "RMSA-Net: Robust Multimodal Sentiment Analysis under Missing Modalities and Temporal Misalignment," submitted to *IEEE Access*, 2026.

RMSA-Net is a lightweight BiLSTM model (1.98 M trainable parameters) for multimodal sentiment analysis on frozen BERT text features, COVAREP audio and Facet visual features. It has three optional modules:

- **MMR** (missing-modality reconstruction): when audio or vision is absent, a hard switch replaces its projection with a surrogate generated from text, f̃m = (1 − µm) fm + µm Rm(ft).
- **MMR-rec**: MMR trained with an extra stop-gradient feature-matching loss on samples in which the modality is present (λ3 = 0.1). Without it (**MMR-task**, λ3 = 0) the surrogate learns little about the modality it replaces.
- **CMTA** (single-head text-queried cross-modal attention) and **PSG** (polarity sentiment gate, a salience-weighted pooling of text tokens).

All results are on CMU-MOSI, five seeds (7, 13, 17, 42, 99), with fixed nested test masks per seed and paired two-sided t-tests across seeds.

## Repository layout

| Path | Contents |
|---|---|
| `rmsa_net.py` | Data loading, RMSA-Net (PSG, BiLSTM encoders, CMTA, MMR / MMR-rec, fusion, heads, unimodal heads), baselines (LF-LSTM, TFN, compact MulT), training, the experiment registry and a resumable runner. Calls `analyze.py` at the end. |
| `analyze.py` | Builds the single-experiment tables, paired statistics and Figures 2, 4, 5 and 6 from `per_run.csv`. |
| `make_paper_tables.py` | Builds the tables that combine several experiments (Tables 4, 4b, 5, 8) and Figure 3. |
| `run_tfrnet.py` | Official TFR-Net baseline through the MMSA toolkit, same data file and seeds. |
| `notebooks/RMSA_Net_Colab.ipynb` | End-to-end run on a Colab T4 GPU with results saved to Google Drive (resumable). |
| `results/per_run.csv` | One row per (configuration, seed, test condition): 1,620 rows, all metrics, fidelity and attention statistics. |
| `results/stats.csv` | Paired t-tests used in the paper. |
| `results/tfrnet/` | TFR-Net result per (frame-loss rate, seed). |
| `results/tables/`, `results/paper_tables/` | Every table as CSV (see the mapping below). |
| `results/*.png` | Figures 2–6. |
| `results/data_info.json` | Feature dimensions, sequence lengths, all-zero stream counts and constant-predictor MAE. |
| `results/psg_analysis.md` | PSG sign-flip check and the correlation of \|p\| with VADER lexicon intensity. |

## Where each result in the paper comes from

`analyze.py` numbers its output files in the order the experiments were run, which differs from the paper. Use this table:

| Paper | File |
|---|---|
| Table 2 (hyperparameters) | `results/tables/table2.csv` |
| Table 3 (main results) | `results/tables/table3.csv` |
| Table 3, continued (published results) | `results/tables/table3_published.csv` |
| Table 4 (re-trained baselines) | `results/paper_tables/table4_baselines.csv` |
| Table 4b (official TFR-Net) | `results/paper_tables/table4b_tfrnet.csv` |
| Table 5 (trained on complete data, streams missing at test) | `results/paper_tables/table5_robustness.csv` |
| Table 6 (surrogate fidelity) | `results/tables/table8.csv` |
| Table 7 (training dropout × test missingness) | `results/tables/table7.csv` |
| Table 8 (partial frame loss and noise) | `results/paper_tables/table8_partial_noise.csv` |
| Table 9 (factorial ablation) | `results/tables/table4.csv` |
| Table 10 (CMTA vs mean-pooling) | `results/tables/table10.csv` |
| Table 10, continued (joint degradation) | `results/tables/table6.csv` |
| Figure 2 (CMTA attention) | `results/fig2_attention.png` |
| Figure 3 (robustness) | `results/fig3_robustness.png` |
| Figure 4 (convergence) | `results/fig4_convergence.png` |
| Figure 5 (ablation) | `results/fig5_ablation.png` |
| Figure 6 (PSG token weights) | `results/fig6_psg.png` |

## Data

CMU-MOSI is not included. Download the MMSA release (`aligned_50.pkl` and `unaligned_50.pkl`) from the links in the [MMSA repository](https://github.com/thuiar/MMSA). It contains 768-d BERT text features with the attention mask (`text_bert`), 5-d COVAREP audio and 20-d Facet visual features; aligned sequences have T = 50, unaligned audio and visual sequences up to 375 and 500 frames.

## Running

```bash
pip install -r requirements.txt

# 1. Quick check on synthetic data (a few minutes on CPU)
python rmsa_net.py --smoke --out /tmp/smoke

# 2. All RMSA-Net and in-pipeline baseline runs (about 4-5 GPU-hours on a T4).
#    Resumable: rerun the same command after an interruption.
python rmsa_net.py --mosi_aligned aligned_50.pkl --mosi_unaligned unaligned_50.pkl \
                   --groups main ablation mmr_analysis unaligned baselines --out results

# 3. Official TFR-Net baseline (fine-tunes BERT; about 4 GPU-hours)
pip install MMSA easydict
python run_tfrnet.py --unaligned unaligned_50.pkl --out results

# 4. Tables and figures
python rmsa_net.py --analyze_only --out results
python make_paper_tables.py --results results
```

Without a GPU, use `notebooks/RMSA_Net_Colab.ipynb`: it writes the three scripts, finds the data in Google Drive and saves every finished run there, so a disconnect loses nothing.

**Checking the reported numbers without training.** Run step 4 on a copy of the `results/` folder:

```bash
cp -r results /tmp/check
python rmsa_net.py --analyze_only --out /tmp/check
python make_paper_tables.py --results /tmp/check
```

This rebuilds every table, `stats.csv` and Figures 3 and 5 from `per_run.csv` and `tfrnet/`. Figures 2, 4 and 6 and the PSG–VADER correlation also need the per-epoch training curves and the per-utterance attention/PSG exports written during training. Those exports contain CMU-MOSI transcripts, so they are not redistributed here; step 2 recreates them.

**TFR-Net note.** With recent PyTorch versions, MMSA's `trains/missingTask/TFR_NET.py` fails on `ReduceLROnPlateau(..., verbose=True)`. Remove `, verbose=True` from that line (the notebook does this automatically).

## Model names in `per_run.csv`

Runs are named `<dataset>|<model>|[rec<λ3>|]tr<p_audio>-<p_visual>`, where the training missingness rates are given last. For RMSA-Net, the three letters after `rmsa-` are PSG (`P` or `-`), CMTA (`C` or `-`) and the imputation mode (`M` = MMR, `Z` = zero-fill, `X` = text-MLP control). `rec0.1` marks MMR-rec. The test condition is in the `test` column: `clean`, `A30`, `V30`, `AV30`, `AV50`, `frame30`, `frame50`, `noise0.5`, `noise1.0`.

## License

MIT (see `LICENSE`). CMU-MOSI is subject to its own license.
