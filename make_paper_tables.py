"""
Builds the tables of the revised paper that combine several experiments (Tables 4, 4b, 5, 8)
and Figure 3, from results/per_run.csv and results/tfrnet/*.json.

    python make_paper_tables.py --results results

All other tables and figures are written by analyze.py (see README for the mapping).
Values are mean ± sample standard deviation over the five seeds.
"""
import argparse, glob, json, os
import numpy as np, pandas as pd

TESTS = [('clean', 'Complete'), ('A30', 'Audio 30%'), ('V30', 'Vision 30%'), ('AV30', 'A+V 30%'), ('AV50', 'A+V 50%')]
DEC = {'mae': 3, 'corr': 3, 'acc7': 2, 'acc2': 2, 'f1': 2}


def ms(x, k):
    x = np.asarray(x, float)
    return f'{x.mean():.{DEC[k]}f} ± {x.std(ddof=1):.{DEC[k]}f}'


def cell(d, run, test, k):
    v = d[(d.run == run) & (d.test == test)][k]
    assert len(v) == 5, (run, test, len(v))
    return ms(v, k)


def main(R):
    d = pd.read_csv(os.path.join(R, 'per_run.csv'))
    out = os.path.join(R, 'paper_tables'); os.makedirs(out, exist_ok=True)
    M = ['mae', 'corr', 'acc7', 'acc2', 'f1']

    # Table 5: models trained on complete data, whole-stream missingness at test
    var = [('Zero-fill', 'mosi|rmsa-PCZ|tr0-0'), ('MMR-task (no reconstruction loss)', 'mosi|rmsa-PCM|tr0-0'),
           ('MMR-rec (feature matching, ours)', 'mosi|rmsa-PCM|rec0.1|tr0-0')]
    rows = [[n, lab] + [cell(d, r, t, k) for k in M[:4]] for n, r in var for t, lab in TESTS]
    pd.DataFrame(rows, columns=['Variant', 'Test condition', 'mae', 'corr', 'acc7', 'acc2']).to_csv(f'{out}/table5_robustness.csv', index=False)

    # Table 4: baselines, matched training dropout
    def p(r): return f"{d[d.run == r].n_params.iloc[0] / 1e6:.2f}"
    bl = [('LF-LSTM', 'lflstm', None), ('TFN [27]', 'tfn', None), ('MulT [6] (re-impl.)', 'mult', 'mult'),
          ('RMSA-Net zero-fill', 'rmsa-PCZ', None), ('RMSA-Net MMR-task', 'rmsa-PCM', 'rmsa-PCM')]
    rows = []
    for name, m, un in bl:
        r = [f'{name} — matched dropout', p(f'mosi|{m}|tr0-0')]
        for tr, te in [('0-0', 'clean'), ('0.3-0.3', 'AV30'), ('0.5-0.5', 'AV50')]:
            r += [cell(d, f'mosi|{m}|tr{tr}', te, 'mae'), cell(d, f'mosi|{m}|tr{tr}', te, 'acc7')]
        r += [cell(d, f'mosi_unal|{un}|tr0-0', 'clean', k) for k in ('mae', 'acc7')] if un else ['—', '—']
        rows.append(r)
    rec = 'mosi|rmsa-PCM|rec0.1|tr0-0'
    rows.append(['RMSA-Net MMR-rec — trained on complete data only', p(rec)] +
                sum([[cell(d, rec, t, 'mae'), cell(d, rec, t, 'acc7')] for t in ('clean', 'AV30', 'AV50')], []) +
                [cell(d, 'mosi_unal|rmsa-PCM|rec0.1|tr0-0', 'clean', k) for k in ('mae', 'acc7')])
    pd.DataFrame(rows, columns=['Model (training regime)', 'Params (M)', 'Complete MAE', 'Complete Acc-7', 'A+V 30% MAE',
                                'A+V 30% Acc-7', 'A+V 50% MAE', 'A+V 50% Acc-7', 'Unaligned MAE', 'Unaligned Acc-7']).to_csv(f'{out}/table4_baselines.csv', index=False)

    # Table 4b: official TFR-Net vs RMSA-Net under frame-level loss (unaligned data)
    tf = pd.DataFrame([json.load(open(f)) for f in glob.glob(os.path.join(R, 'tfrnet', 'rate*_s*.json'))])
    rows = []
    for rate in (0.0, 0.3, 0.5):
        s = tf[tf.rate == rate]; assert len(s) == 5
        rows.append([f'TFR-Net [15], official (BERT fine-tuned, 124.3 M), frame loss {int(rate * 100)}% in training and test'] +
                    [ms(s[k], k) for k in ('mae', 'acc7', 'acc2', 'f1')])
    for name, r in [('MMR-task', 'mosi_unal|rmsa-PCM|tr0-0'), ('MMR-rec', 'mosi_unal|rmsa-PCM|rec0.1|tr0-0')]:
        for i, (t, pct) in enumerate([('clean', 0), ('frame30', 30), ('frame50', 50)]):
            lab = f'RMSA-Net {name} (frozen BERT, 1.98 M), trained complete, frame loss {pct}% at test' if i == 0 else f'… frame loss {pct}% at test'
            rows.append([lab] + [cell(d, r, t, k) for k in ('mae', 'acc7', 'acc2', 'f1')])
    pd.DataFrame(rows, columns=['Model', 'MAE', 'Acc-7', 'Acc-2', 'F1']).to_csv(f'{out}/table4b_tfrnet.csv', index=False)

    # Table 8: partial frame loss and additive noise
    pv = [('Clean-trained MMR', 'mosi|rmsa-PCM|tr0-0'), ('A+V30-trained MMR', 'mosi|rmsa-PCM|tr0.3-0.3'),
          ('A+V30-trained zero-fill', 'mosi|rmsa-PCZ|tr0.3-0.3'), ('A+V30-trained MMR-rec', 'mosi|rmsa-PCM|rec0.1|tr0.3-0.3'),
          ('Clean-trained MMR-rec', 'mosi|rmsa-PCM|rec0.1|tr0-0')]
    rows = [[f'{n} @ {t}'] + [cell(d, r, t, k) for k in M] + [5] for n, r in pv for t in ('clean', 'frame50', 'noise1.0')]
    pd.DataFrame(rows, columns=['Condition', 'MAE↓', 'Corr↑', 'Acc-7↑', 'Acc-2↑', 'F1↑', 'n seeds']).to_csv(f'{out}/table8_partial_noise.csv', index=False)

    # Figure 3
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for (name, r), mk, c in zip(var, 's^o', ['#e07b39', '#c44e52', '#4c72b0']):
        for a, k in zip(ax, ('mae', 'acc7')):
            v = [d[(d.run == r) & (d.test == t)][k] for t, _ in TESTS]
            a.errorbar(range(5), [x.mean() for x in v], [x.std(ddof=1) for x in v], marker=mk, color=c, capsize=3, lw=1.8, label=name)
    for a, y in zip(ax, ('MAE (lower is better)', 'Acc-7 (%)')):
        a.set_xticks(range(5)); a.set_xticklabels([l for _, l in TESTS], fontsize=8); a.set_ylabel(y); a.grid(alpha=.3)
    ax[0].legend(fontsize=8)
    fig.suptitle('Models trained on complete inputs, tested with whole audio/visual streams missing (CMU-MOSI, mean ± std, 5 seeds)', fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(R, 'fig3_robustness.png'), dpi=200)
    print('wrote', out, 'and fig3_robustness.png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--results', default='results'); main(ap.parse_args().results)
