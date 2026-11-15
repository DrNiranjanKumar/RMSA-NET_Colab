"""
TFR-Net baseline (R1.4, R2.4) with the official implementation in the MMSA toolkit, on the same CMU-MOSI
unaligned file and the same five seeds as RMSA-Net. Resume-safe: one JSON per (rate, seed) in OUT/tfrnet/.

TFR-Net's own missing-data protocol drops random FRAMES (not whole streams) at the given rate, in training and
test, and fine-tunes BERT. Text is kept complete (rate 0) so the setting matches RMSA-Net's audio/visual
degradation as closely as the official code allows; compare it with the frame30 rows of Table 9.

    pip install MMSA
    python run_tfrnet.py --unaligned /content/data/unaligned_50.pkl --out /content/drive/MyDrive/rmsa_results_v2
"""
import argparse, ast, json, os, re, time


def parse_cell(v):
    """MMSA writes '(mean*100, std*100)' per metric; newer NumPy writes '(np.float64(79.97), np.float64(0.0))'."""
    if isinstance(v, str):
        v = ast.literal_eval(re.sub(r"np\.(float|int)\d*\(([^()]*)\)", r"\2", v))
    return float(v[0]) if isinstance(v, (tuple, list)) else float(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unaligned", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[13, 42, 7, 17, 99])
    ap.add_argument("--rates", nargs="+", type=float, default=[0.0, 0.3, 0.5])
    a = ap.parse_args()
    import pandas as pd
    from MMSA import MMSA_run
    base = os.path.join(a.out, "tfrnet"); os.makedirs(base, exist_ok=True)
    for r in a.rates:
        for s in a.seeds:
            dst = os.path.join(base, f"rate{r:g}_s{s}.json")
            if os.path.exists(dst):
                print("done:", dst); continue
            raw = os.path.join(base, f"raw_rate{r:g}_s{s}")
            csv = os.path.join(raw, "normal", "mosi.csv")
            t0 = time.time()
            if os.path.exists(csv):
                print("reusing finished MMSA result:", csv)
            else:
              MMSA_run("tfr_net", "mosi", seeds=[s], gpu_ids=[0], num_workers=2,
                     config={"featurePath": a.unaligned, "missing_rate": [0.0, r, r], "missing_seed": [1111 + s] * 3},
                     res_save_dir=raw, model_save_dir="/content/mmsa_models", log_dir=raw)
            row = pd.read_csv(csv).iloc[-1]
            res = dict(model="tfr_net", rate=r, seed=s, train_sec=round(time.time() - t0, 1),
                       mae=parse_cell(row["MAE"]) / 100, corr=parse_cell(row["Corr"]) / 100,
                       acc7=parse_cell(row["Mult_acc_7"]), acc2=parse_cell(row["Has0_acc_2"]),
                       f1=parse_cell(row["Has0_F1_score"]), acc2_nz=parse_cell(row["Non0_acc_2"]),
                       f1_nz=parse_cell(row["Non0_F1_score"]))
            json.dump(res, open(dst, "w"))
            print(res)


if __name__ == "__main__":
    main()
