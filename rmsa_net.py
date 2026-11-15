"""
RMSA-Net: re-implementation of the architecture described in Section III of the
IEEE Access manuscript, plus every experiment requested by the reviewers.

One file: data loading, model (Eqs. 1-28), baselines (TFN, LF-LSTM, MulT-compact),
training (Eqs. 29-31), the experiment registry, and analysis hooks.

Usage (Colab):
    python rmsa_net.py --mosi_aligned /content/drive/MyDrive/MOSI/aligned_50.pkl \
                       --mosi_unaligned /content/drive/MyDrive/MOSI/unaligned_50.pkl \
                       --groups main ablation mmr_analysis unaligned baselines \
                       --out /content/drive/MyDrive/rmsa_results
    python analyze.py --out /content/drive/MyDrive/rmsa_results

Smoke test on synthetic data (CPU, ~1 minute):
    python rmsa_net.py --smoke --out /tmp/rmsa_smoke
"""
from __future__ import annotations

import argparse, json, math, os, pickle, time, random
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================================
# 1. Reproducibility
# =====================================================================
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =====================================================================
# 2. Data
# =====================================================================
def _first(d: dict, keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _compact(x: np.ndarray, lengths: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Return sequences with valid (non-zero) frames moved to the front, plus lengths.
    If the file provides lengths (MMSA format, padding at the end) they are used as-is."""
    x = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if lengths is not None:
        return x, np.clip(np.asarray(lengths).astype(np.int64), 1, x.shape[1])
    valid = np.abs(x).sum(-1) > 0                                # (N, T)
    order = np.argsort(~valid, axis=1, kind="stable")             # valid rows first
    x = np.take_along_axis(x, order[..., None], axis=1)
    return x, np.clip(valid.sum(1), 1, x.shape[1])


def load_split_file(path: str) -> Dict[str, dict]:
    """Loads CMU-MOSI / MOSEI pickles in either the MMSA format
    (keys: text/audio/vision/regression_labels[/audio_lengths/vision_lengths/raw_text])
    or the MulT format (keys: text/audio/vision/labels)."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    out = {}
    for split, alias in [("train", ["train"]), ("valid", ["valid", "dev", "val"]), ("test", ["test"])]:
        raw = _first(data, alias)
        if raw is None:
            raise KeyError(f"{path}: no split {alias}")
        text = _first(raw, ["text", "glove", "text_glove"])
        audio = _first(raw, ["audio", "covarep"])
        vision = _first(raw, ["vision", "facet", "openface"])
        labels = _first(raw, ["regression_labels", "labels"])
        if text is None or audio is None or vision is None or labels is None:
            raise KeyError(f"{path}/{split}: missing text/audio/vision/labels; keys = {list(raw.keys())}")
        if text.ndim != 3:
            raise ValueError(f"{path}: text must be (N,T,d); got {text.shape}. "
                             "Use a release with GloVe (300-d) or BERT feature sequences, not token ids.")
        # MMSA releases carry BERT token ids + attention mask in text_bert (N, 3, T). Use the mask for the true
        # text length: BERT features at [PAD] positions are non-zero, so zero-row detection would count padding.
        tb = _first(raw, ["text_bert"])
        ids, tl_given = None, None
        if tb is not None and np.asarray(tb).ndim == 3 and np.asarray(tb).shape[1] == 3 \
                and np.asarray(tb).shape[2] == np.asarray(text).shape[1]:
            tb = np.asarray(tb)
            ids, tl_given = tb[:, 0, :].astype(np.int64), tb[:, 1, :].sum(1)
        t, tl = _compact(np.asarray(text), tl_given)
        a, al = _compact(np.asarray(audio), _first(raw, ["audio_lengths"]))
        v, vl = _compact(np.asarray(vision), _first(raw, ["vision_lengths"]))
        y = np.clip(np.asarray(labels, dtype=np.float32).reshape(-1), -3, 3)
        out[split] = dict(text=t, audio=a, vision=v, tl=tl, al=al, vl=vl, y=y, text_ids=ids,
                          raw_text=_first(raw, ["raw_text"]), ids=_first(raw, ["id", "ids"]))
    # Count segments where a modality is entirely missing in the file itself (R1.7: report this)
    out["_stats"] = {s: {"n": int(len(out[s]["y"])),
                         "audio_all_zero": int((np.abs(out[s]["audio"]).sum((1, 2)) == 0).sum()),
                         "vision_all_zero": int((np.abs(out[s]["vision"]).sum((1, 2)) == 0).sum()),
                         "T_text": int(out[s]["text"].shape[1]), "text_len_mean": float(out[s]["tl"].mean()),
                         "text_len_source": "text_bert attention mask" if out[s]["text_ids"] is not None else "non-zero rows", "T_audio": int(out[s]["audio"].shape[1]),
                         "T_vision": int(out[s]["vision"].shape[1]),
                         "d": [int(out[s]["text"].shape[2]), int(out[s]["audio"].shape[2]), int(out[s]["vision"].shape[2])]}
                     for s in ["train", "valid", "test"]}
    return out


def standardize(ds: Dict[str, dict]):
    """z-score audio and vision with training statistics over valid frames (text untouched)."""
    for m, lk in [("audio", "al"), ("vision", "vl")]:
        x, l = ds["train"][m], ds["train"][lk]
        mask = np.arange(x.shape[1])[None, :] < l[:, None]
        frames = x[mask]
        frames = frames[np.abs(frames).sum(-1) > 0] if len(frames) else frames
        mu, sd = frames.mean(0), frames.std(0) + 1e-6
        for s in ["train", "valid", "test"]:
            X = ds[s][m]; L = ds[s][lk]
            msk = (np.arange(X.shape[1])[None, :] < L[:, None]) & (np.abs(X).sum(-1) > 0)
            X[msk] = (X[msk] - mu) / sd
        ds["_std_" + m] = float(sd.mean())


def synthetic_dataset(n=(300, 60, 120), T=(20, 60, 40), d=(300, 5, 20), seed=0) -> Dict[str, dict]:
    """Small synthetic data set with a learnable signal; used only for the smoke test."""
    rng = np.random.RandomState(seed)
    out = {}
    w = [rng.randn(dd) / math.sqrt(dd) for dd in d]
    for split, N in zip(["train", "valid", "test"], n):
        y = np.clip(rng.randn(N) * 1.5, -3, 3).astype(np.float32)
        feats, lens = [], []
        for k, (TT, dd) in enumerate(zip(T, d)):
            L = rng.randint(TT // 2, TT + 1, size=N)
            x = rng.randn(N, TT, dd).astype(np.float32) * 0.5
            x += (y[:, None, None] * w[k][None, None, :] * (1.5 if k == 0 else 0.6)).astype(np.float32)
            x[np.arange(TT)[None, :] >= L[:, None]] = 0
            feats.append(x); lens.append(L)
        vocab = ["good", "bad", "great", "terrible", "movie", "the", "plot", "was", "really", "boring", "love", "it"]
        raw = np.array([" ".join(rng.choice(vocab, int(L))) for L in lens[0]])
        out[split] = dict(text=feats[0], audio=feats[1], vision=feats[2], tl=lens[0], al=lens[1], vl=lens[2],
                          y=y, raw_text=raw, ids=None)
    out["_stats"] = {}
    return out


@dataclass
class Degrade:
    """A test (or training) degradation condition."""
    name: str = "clean"
    pa: float = 0.0          # P(audio stream entirely missing)
    pv: float = 0.0          # P(vision stream entirely missing)
    frame_drop: float = 0.0  # fraction of frames zeroed inside present a/v streams (partial loss)
    noise: float = 0.0       # Gaussian noise std (in standardised feature units) added to present a/v streams


class Batcher:
    """Yields padded mini-batches with missing-modality simulation.

    Training masks are re-drawn every epoch (Eq. 29). Evaluation masks are fixed per seed:
    uniforms U_a, U_v ~ U(0,1) are drawn once from mask_seed and mu_m = 1[U_m < p_m], so the
    masks for different rates are nested and identical across all conditions of one seed
    (paired comparisons see the same test mask)."""

    def __init__(self, split: dict, batch_size: int, mask_seed: int):
        self.s = split
        self.N = len(split["y"])
        self.bs = batch_size
        r = np.random.RandomState(mask_seed)
        self.Ua, self.Uv = r.rand(self.N), r.rand(self.N)
        self.frame_rng_seed = mask_seed + 7

    def _apply(self, idx, deg: Degrade, train: bool, gen: Optional[np.random.RandomState]):
        s = self.s
        text = torch.from_numpy(s["text"][idx]); audio = torch.from_numpy(s["audio"][idx].copy())
        vision = torch.from_numpy(s["vision"][idx].copy())
        tl = torch.from_numpy(s["tl"][idx]); al = torch.from_numpy(s["al"][idx].copy()); vl = torch.from_numpy(s["vl"][idx].copy())
        if train:
            mu_a = torch.from_numpy((gen.rand(len(idx)) < deg.pa).astype(np.float32))
            mu_v = torch.from_numpy((gen.rand(len(idx)) < deg.pv).astype(np.float32))
        else:
            mu_a = torch.from_numpy((self.Ua[idx] < deg.pa).astype(np.float32))
            mu_v = torch.from_numpy((self.Uv[idx] < deg.pv).astype(np.float32))
        frng = gen if train else np.random.RandomState(self.frame_rng_seed + int(idx[0]))
        for x, L, mu in [(audio, al, mu_a), (vision, vl, mu_v)]:
            if deg.frame_drop > 0:
                drop = torch.from_numpy(frng.rand(*x.shape[:2]) < deg.frame_drop)
                x[drop] = 0.0
            if deg.noise > 0:
                valid = (torch.arange(x.shape[1])[None, :] < L[:, None]).float()[..., None]
                x += torch.from_numpy(frng.randn(*x.shape).astype(np.float32)) * deg.noise * valid
            x[mu.bool()] = 0.0                       # whole stream missing -> all-zero tensor
            L[mu.bool()] = 1
        return dict(text=text, audio=audio, vision=vision, tl=tl, al=al, vl=vl,
                    y=torch.from_numpy(s["y"][idx]), idx=torch.from_numpy(idx))

    def iterate(self, deg: Degrade, train: bool, gen: Optional[np.random.RandomState] = None):
        order = gen.permutation(self.N) if train else np.arange(self.N)
        for i in range(0, self.N, self.bs):
            idx = order[i:i + self.bs]
            b = self._apply(idx, deg, train, gen)
            yield {k: v.to(DEVICE) for k, v in b.items()}


# =====================================================================
# 3. Model (Section III)
# =====================================================================
def lengths_mask(lengths: torch.Tensor, T: int) -> torch.Tensor:
    return torch.arange(T, device=lengths.device)[None, :] < lengths[:, None]


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.float().unsqueeze(-1)
    return (x * m).sum(1) / m.sum(1).clamp(min=1.0)


class BiLSTMEncoder(nn.Module):
    """Eqs. (1)-(3). hidden = per-direction size; output dim = 2*hidden."""
    def __init__(self, d_in, hidden, layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(d_in, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                            dropout=dropout if layers > 1 else 0.0)

    def forward(self, x, lengths):
        T = x.shape[1]
        packed = pack_padded_sequence(x, lengths.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=T)
        return out


class PSG(nn.Module):
    """Polarity Sentiment Gate, Eqs. (13)-(17).
    G = sigma(Wg2 tanh(Wg1 X + b) + b);  X~ = X * G;  p = tanh(Wp2 ReLU(Wp1 X~ + b) + b);
    alpha = softmax(|p|) over valid tokens;  c_t = sum_i alpha_i X~_i."""
    def __init__(self, d_t, H):
        super().__init__()
        self.g1, self.g2 = nn.Linear(d_t, H), nn.Linear(H, d_t)
        self.p1, self.p2 = nn.Linear(d_t, H), nn.Linear(H, 1)

    def forward(self, X, tmask):
        G = torch.sigmoid(self.g2(torch.tanh(self.g1(X))))
        Xg = X * G
        p = torch.tanh(self.p2(F.relu(self.p1(Xg)))).squeeze(-1)          # (B,T)
        score = p.abs().masked_fill(~tmask, float("-inf"))
        alpha = torch.softmax(score, dim=1)
        c = (alpha.unsqueeze(-1) * Xg).sum(1)
        return Xg, c, alpha, p


class CMTA(nn.Module):
    """Cross-Modal Temporal Attention, Eqs. (7)-(12): single head, text queries,
    audio/visual keys/values at native length, padding masked, mean-pooled over valid queries."""
    def __init__(self, d_q, d_k, H):
        super().__init__()
        self.Wq, self.Wk, self.Wv = nn.Linear(d_q, H, bias=False), nn.Linear(d_k, H, bias=False), nn.Linear(d_k, H, bias=False)
        self.H = H

    def forward(self, Hq, qmask, Hk, kmask):
        Q, K, V = self.Wq(Hq), self.Wk(Hk), self.Wv(Hk)
        s = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.H)
        s = s.masked_fill(~kmask[:, None, :], float("-inf"))
        A = torch.softmax(s, dim=-1)
        out = torch.bmm(A, V)
        return masked_mean(out, qmask), A


class RMSANet(nn.Module):
    """mmr_mode: 'mmr'     hard-switch imputation, Eq. (6)
                 'zero'    zero-fill: absent projection is used as-is (no surrogate)
                 'textmlp' control: zero-fill + an always-on text MLP branch R_m(f_t) concatenated to the fusion input
                           (same extra parameters as MMR, no switch) -- tests whether MMR gains are just extra text capacity."""
    def __init__(self, d_t, d_a, d_v, H=128, D=64, dropout=0.3,
                 use_psg=True, use_cmta=True, mmr_mode="mmr"):
        super().__init__()
        self.use_psg, self.use_cmta, self.mmr_mode = use_psg, use_cmta, mmr_mode
        self.psg = PSG(d_t, H) if use_psg else None
        self.enc_t = BiLSTMEncoder(d_t, H, 2, dropout)              # -> 2H = 256
        self.enc_a = BiLSTMEncoder(d_a, H // 2, 1, 0.0)             # -> H = 128
        self.enc_v = BiLSTMEncoder(d_v, H // 2, 1, 0.0)
        if use_cmta:
            self.cmta_a, self.cmta_v = CMTA(2 * H, H, H), CMTA(2 * H, H, H)
        self.proj_t = nn.Linear(2 * H + d_t, D)                      # Eq. (18), W_t in R^{D x 556}
        self.proj_a = nn.Linear(2 * H, D)                            # Eq. (19)
        self.proj_v = nn.Linear(2 * H, D)                            # Eq. (20)
        self.R_a = nn.Sequential(nn.Linear(D, H), nn.ReLU(), nn.Linear(H, D))   # Eq. (5)
        self.R_v = nn.Sequential(nn.Linear(D, H), nn.ReLU(), nn.Linear(H, D))
        n_in = 5 * D if mmr_mode == "textmlp" else 3 * D
        self.fuse = nn.Sequential(nn.Linear(n_in, H), nn.ReLU(), nn.Dropout(dropout), nn.Linear(H, D // 2), nn.ReLU())  # Eq. (21)
        self.head_reg, self.head_cls = nn.Linear(D // 2, 1), nn.Linear(D // 2, 1)          # Eqs. (22)-(23)
        self.uni = nn.ModuleDict({m: nn.Linear(D, 1) for m in "tav"})                       # Eq. (24)

    def forward(self, b):
        X, Xa, Xv = b["text"], b["audio"], b["vision"]
        tmask = lengths_mask(b["tl"], X.shape[1])
        amask = lengths_mask(b["al"], Xa.shape[1])
        vmask = lengths_mask(b["vl"], Xv.shape[1])
        # Eq. (4): absence indicator from the input itself
        mu_a = (Xa.abs().sum((1, 2)) == 0).float()
        mu_v = (Xv.abs().sum((1, 2)) == 0).float()

        if self.use_psg:
            Xg, c_t, alpha, p = self.psg(X, tmask)
        else:
            Xg, c_t, alpha, p = X, masked_mean(X, tmask), None, None
        Ht = self.enc_t(Xg, b["tl"]); Ha = self.enc_a(Xa, b["al"]); Hv = self.enc_v(Xv, b["vl"])
        Ht_bar, Ha_bar, Hv_bar = masked_mean(Ht, tmask), masked_mean(Ha, amask), masked_mean(Hv, vmask)
        if self.use_cmta:
            h_ta, A_ta = self.cmta_a(Ht, tmask, Ha, amask)
            h_tv, A_tv = self.cmta_v(Ht, tmask, Hv, vmask)
        else:
            h_ta, h_tv, A_ta, A_tv = Ha_bar, Hv_bar, None, None
        f_t = F.relu(self.proj_t(torch.cat([Ht_bar, c_t], -1)))
        f_a = F.relu(self.proj_a(torch.cat([Ha_bar, h_ta], -1)))
        f_v = F.relu(self.proj_v(torch.cat([Hv_bar, h_tv], -1)))
        fh_a, fh_v = self.R_a(f_t), self.R_v(f_t)

        if self.mmr_mode == "mmr":                                     # Eq. (6)
            ft_a = (1 - mu_a)[:, None] * f_a + mu_a[:, None] * fh_a
            ft_v = (1 - mu_v)[:, None] * f_v + mu_v[:, None] * fh_v
            fused_in = torch.cat([f_t, ft_a, ft_v], -1)
        elif self.mmr_mode == "zero":
            fused_in = torch.cat([f_t, f_a, f_v], -1)
        elif self.mmr_mode == "textmlp":
            fused_in = torch.cat([f_t, f_a, f_v, fh_a, fh_v], -1)
        else:
            raise ValueError(self.mmr_mode)
        z = self.fuse(fused_in)
        return dict(y_reg=self.head_reg(z).squeeze(-1), y_cls=self.head_cls(z).squeeze(-1),
                    y_uni={m: self.uni[m](f).squeeze(-1) for m, f in zip("tav", [f_t, f_a, f_v])},
                    f_t=f_t, f_a=f_a, f_v=f_v, fh_a=fh_a, fh_v=fh_v, mu_a=mu_a, mu_v=mu_v,
                    A_ta=A_ta, A_tv=A_tv, alpha=alpha, p=p, tmask=tmask)


# ---------------------------------------------------------------------
# Baselines, trained in the same pipeline (same features, masks, seeds)
# ---------------------------------------------------------------------
class LFLSTM(nn.Module):
    """Late-fusion BiLSTM: per-modality BiLSTM, mean-pool, concat, MLP. Zero-fill for missing input."""
    def __init__(self, d_t, d_a, d_v, H=128, dropout=0.3):
        super().__init__()
        self.e = nn.ModuleList([BiLSTMEncoder(d, H // 2, 1, 0.0) for d in (d_t, d_a, d_v)])
        self.mlp = nn.Sequential(nn.Linear(3 * H, H), nn.ReLU(), nn.Dropout(dropout), nn.Linear(H, 1))

    def forward(self, b):
        hs = []
        for enc, x, l in zip(self.e, (b["text"], b["audio"], b["vision"]), (b["tl"], b["al"], b["vl"])):
            hs.append(masked_mean(enc(x, l), lengths_mask(l, x.shape[1])))
        return dict(y_reg=self.mlp(torch.cat(hs, -1)).squeeze(-1))


class TFN(nn.Module):
    """Tensor Fusion Network (Zadeh et al., 2017): text LSTM subnet, mean-pooled audio/visual DNN subnets,
    3-fold outer product with appended 1s, post-fusion MLP."""
    def __init__(self, d_t, d_a, d_v, ht=64, ha=16, hv=16, post=128, dropout=0.3):
        super().__init__()
        self.text = BiLSTMEncoder(d_t, ht // 2, 1, 0.0)
        self.audio = nn.Sequential(nn.Linear(d_a, ha), nn.ReLU(), nn.Linear(ha, ha), nn.ReLU())
        self.vision = nn.Sequential(nn.Linear(d_v, hv), nn.ReLU(), nn.Linear(hv, hv), nn.ReLU())
        self.post = nn.Sequential(nn.Dropout(dropout), nn.Linear((ht + 1) * (ha + 1) * (hv + 1), post), nn.ReLU(),
                                  nn.Linear(post, post), nn.ReLU(), nn.Linear(post, 1))

    def forward(self, b):
        zt = masked_mean(self.text(b["text"], b["tl"]), lengths_mask(b["tl"], b["text"].shape[1]))
        za = self.audio(masked_mean(b["audio"], lengths_mask(b["al"], b["audio"].shape[1])))
        zv = self.vision(masked_mean(b["vision"], lengths_mask(b["vl"], b["vision"].shape[1])))
        one = torch.ones(zt.shape[0], 1, device=zt.device)
        zt, za, zv = torch.cat([zt, one], 1), torch.cat([za, one], 1), torch.cat([zv, one], 1)
        fusion = torch.einsum("bi,bj,bk->bijk", zt, za, zv).flatten(1)
        return dict(y_reg=self.post(fusion).squeeze(-1))


def sinusoid(T, d, device):
    pos = torch.arange(T, device=device).float()[:, None]
    i = torch.arange(d, device=device).float()[None, :]
    ang = pos / torch.pow(10000, (2 * (i // 2)) / d)
    return torch.where(i % 2 == 0, torch.sin(ang), torch.cos(ang))


class XBlock(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.ln_q, self.ln_k, self.ln_f = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(4 * d, d))

    def forward(self, q, k, kpad):
        kk = self.ln_k(k)
        q = q + self.att(self.ln_q(q), kk, kk, key_padding_mask=kpad, need_weights=False)[0]
        return q + self.ff(self.ln_f(q))


class MulTCompact(nn.Module):
    """Compact re-implementation of MulT (Tsai et al., 2019): 1-D conv projections, six directed
    cross-modal transformers, per-target self-attention, residual MLP head. Masked mean-pooling replaces
    'last time step' so that variable-length padded sequences are handled. Defaults follow MulT's
    CMU-MOSI setting (d = 30, 5 heads, 4 layers)."""
    def __init__(self, d_t, d_a, d_v, d=30, heads=5, layers=4, dropout=0.1):
        super().__init__()
        self.proj = nn.ModuleList([nn.Conv1d(di, d, 1, bias=False) for di in (d_t, d_a, d_v)])
        self.cross = nn.ModuleDict({f"{t}{s}": nn.ModuleList([XBlock(d, heads, dropout) for _ in range(layers)])
                                    for t in "tav" for s in "tav" if t != s})
        self.mem = nn.ModuleDict({t: nn.ModuleList([XBlock(2 * d, heads, dropout) for _ in range(3)]) for t in "tav"})
        self.out = nn.Sequential(nn.Linear(6 * d, 6 * d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(6 * d, 6 * d))
        self.head = nn.Linear(6 * d, 1)

    def forward(self, b):
        xs = {m: self.proj[i](b[k].transpose(1, 2)).transpose(1, 2) for i, (m, k) in enumerate(zip("tav", ("text", "audio", "vision")))}
        L = {"t": b["tl"], "a": b["al"], "v": b["vl"]}
        masks = {m: lengths_mask(L[m], xs[m].shape[1]) for m in "tav"}
        xs = {m: x + sinusoid(x.shape[1], x.shape[2], x.device)[None] for m, x in xs.items()}
        hs = []
        for t in "tav":
            outs = []
            for s in "tav":
                if s == t:
                    continue
                h = xs[t]
                for blk in self.cross[f"{t}{s}"]:
                    h = blk(h, xs[s], ~masks[s])
                outs.append(h)
            h = torch.cat(outs, -1)
            for blk in self.mem[t]:
                h = blk(h, h, ~masks[t])
            hs.append(masked_mean(h, masks[t]))
        h = torch.cat(hs, -1)
        return dict(y_reg=self.head(h + self.out(h)).squeeze(-1))


# =====================================================================
# 4. Loss (Eqs. 25-28 and the optional feature-matching term 25b)
# =====================================================================
def rmsa_loss(out, y, lam1=0.5, lam2=0.1, lam3=0.0, mask_uni=True):
    L = F.mse_loss(out["y_reg"], y)
    if "y_cls" not in out:
        return L, {}
    L_cls = F.binary_cross_entropy_with_logits(out["y_cls"], (y >= 0).float())
    L_uni = 0.0
    for m in "tav":
        e = (out["y_uni"][m] - y) ** 2
        if mask_uni and m != "t":
            keep = 1 - out[f"mu_{m}"]
            e = (e * keep).sum() / keep.sum().clamp(min=1)
        else:
            e = e.mean()
        L_uni = L_uni + e
    L = L + lam1 * L_cls + lam2 * L_uni
    L_rec = torch.tensor(0.0, device=y.device)
    if lam3 > 0:
        for m in "av":
            keep = 1 - out[f"mu_{m}"]
            se = ((out[f"fh_{m}"] - out[f"f_{m}"].detach()) ** 2).sum(-1)       # stop-gradient on target
            L_rec = L_rec + (se * keep).sum() / keep.sum().clamp(min=1)
        L = L + lam3 * L_rec
    return L, {"rec": float(L_rec.detach())}


# =====================================================================
# 5. Metrics
# =====================================================================
def metrics(pred: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import f1_score
    pred = np.clip(pred, -3, 3)
    mae = float(np.abs(pred - y).mean())
    corr = float(np.corrcoef(pred, y)[0, 1]) if np.std(pred) > 1e-8 else 0.0
    acc7 = float((np.clip(np.round(pred), -3, 3) == np.clip(np.round(y), -3, 3)).mean() * 100)
    pb, yb = pred >= 0, y >= 0                                       # paper convention (Eq. 34)
    acc2 = float((pb == yb).mean() * 100)
    f1 = float(f1_score(yb, pb, average="weighted") * 100)
    nz = y != 0                                                      # common 'non-zero' convention
    acc2_nz = float(((pred[nz] > 0) == (y[nz] > 0)).mean() * 100)
    f1_nz = float(f1_score(y[nz] > 0, pred[nz] > 0, average="weighted") * 100)
    return dict(mae=mae, corr=corr, acc7=acc7, acc2=acc2, f1=f1, acc2_nz=acc2_nz, f1_nz=f1_nz)


@torch.no_grad()
def evaluate(model, batcher: Batcher, deg: Degrade, extras=False) -> Dict[str, float]:
    model.eval()
    preds, ys = [], []
    cos = {"a": [], "v": []}; mse = {"a": [], "v": []}
    ent = {"a": [], "v": []}
    for b in batcher.iterate(deg, train=False):
        out = model(b)
        preds.append(out["y_reg"].cpu().numpy()); ys.append(b["y"].cpu().numpy())
        if extras and "fh_a" in out:
            for m in "av":
                pres = out[f"mu_{m}"] == 0
                if pres.any():
                    cos[m].append(F.cosine_similarity(out[f"fh_{m}"][pres], out[f"f_{m}"][pres], dim=-1).cpu().numpy())
                    mse[m].append(((out[f"fh_{m}"][pres] - out[f"f_{m}"][pres]) ** 2).sum(-1).cpu().numpy())
                A = out.get(f"A_t{m}")
                if A is not None and pres.any():
                    L = b[f"{m}l"][pres].float()
                    Ap = A[pres].clamp(min=1e-12)
                    H = -(Ap * Ap.log()).sum(-1)                        # (B',Tq)
                    qm = out["tmask"][pres].float()
                    Hn = (H / L.log().clamp(min=1e-6)[:, None] * qm).sum(1) / qm.sum(1)   # entropy / log(T_k): 1 = uniform
                    ent[m].append(Hn.cpu().numpy())
    r = metrics(np.concatenate(preds), np.concatenate(ys))
    if extras:
        for m in "av":
            r[f"fid_cos_{m}"] = float(np.concatenate(cos[m]).mean()) if cos[m] else float("nan")
            r[f"fid_mse_{m}"] = float(np.concatenate(mse[m]).mean()) if mse[m] else float("nan")
            r[f"attn_entropy_{m}"] = float(np.concatenate(ent[m]).mean()) if ent[m] else float("nan")
    return r


# =====================================================================
# 6. Experiment configuration and registry
# =====================================================================
@dataclass
class RunCfg:
    dataset: str = "mosi"            # key into the loaded data sets
    model: str = "rmsa"              # rmsa | tfn | lflstm | mult
    use_psg: bool = True
    use_cmta: bool = True
    mmr_mode: str = "mmr"            # mmr | zero | textmlp
    lam3: float = 0.0                # 0 = MMR-task (paper); >0 = MMR-rec
    train: Degrade = field(default_factory=Degrade)
    group: str = ""

    @property
    def name(self) -> str:
        tr = f"tr{self.train.pa:g}-{self.train.pv:g}"
        if self.model != "rmsa":
            return f"{self.dataset}|{self.model}|{tr}"
        flags = ("P" if self.use_psg else "-") + ("C" if self.use_cmta else "-") + {"mmr": "M", "zero": "Z", "textmlp": "X"}[self.mmr_mode]
        rec = f"|rec{self.lam3:g}" if self.lam3 > 0 else ""
        return f"{self.dataset}|rmsa-{flags}{rec}|{tr}"


# Test conditions evaluated for EVERY trained model (inference only, cheap).
TESTS = [Degrade("clean"), Degrade("A30", 0.3, 0), Degrade("V30", 0, 0.3), Degrade("AV30", 0.3, 0.3),
         Degrade("AV50", 0.5, 0.5), Degrade("frame30", frame_drop=0.3), Degrade("frame50", frame_drop=0.5), Degrade("noise0.5", noise=0.5),
         Degrade("noise1.0", noise=1.0)]
TR = {"clean": Degrade("clean"), "A30": Degrade("A30", 0.3, 0), "V30": Degrade("V30", 0, 0.3),
      "AV30": Degrade("AV30", 0.3, 0.3), "AV50": Degrade("AV50", 0.5, 0.5)}


def registry(groups: List[str]) -> List[RunCfg]:
    R: List[RunCfg] = []
    if "main" in groups:              # Table 3 (aligned) and Table 7 (cross-evaluation comes for free)
        R.append(RunCfg(train=TR["clean"], group="main"))
        for c in ["A30", "V30", "AV30", "AV50"]:
            R.append(RunCfg(train=TR[c], group="main"))
            R.append(RunCfg(train=TR[c], mmr_mode="zero", group="main"))
        R.append(RunCfg(train=TR["clean"], mmr_mode="zero", group="main"))
    if "ablation" in groups:          # Table 4: full factorial under A+V 30%
        for psg in (False, True):
            for cmta in (False, True):
                for mm in ("zero", "mmr"):
                    R.append(RunCfg(train=TR["AV30"], use_psg=psg, use_cmta=cmta, mmr_mode=mm, group="ablation"))
    if "mmr_analysis" in groups:      # Table 8: what the surrogate encodes
        R.append(RunCfg(train=TR["AV30"], lam3=0.1, group="mmr_analysis"))
        R.append(RunCfg(train=TR["AV30"], mmr_mode="textmlp", group="mmr_analysis"))
        R.append(RunCfg(train=TR["clean"], lam3=0.1, group="mmr_analysis"))   # MMR-rec can learn R_m even without dropout
        R.append(RunCfg(dataset="mosi_unal", train=TR["clean"], lam3=0.1, group="mmr_analysis"))   # vs TFR-Net (Table 3c)
    if "unaligned" in groups:         # Table 3 unaligned row, Table 6 joint degradation
        R.append(RunCfg(dataset="mosi_unal", train=TR["clean"], group="unaligned"))
        R.append(RunCfg(dataset="mosi_unal", train=TR["clean"], use_cmta=False, group="unaligned"))
        R.append(RunCfg(train=TR["clean"], use_cmta=False, group="unaligned"))            # aligned, no CMTA (Table 10)
        R.append(RunCfg(dataset="mosi_unal", train=TR["AV30"], group="unaligned"))
        R.append(RunCfg(dataset="mosi_unal", train=TR["AV30"], mmr_mode="zero", group="unaligned"))
    if "baselines" in groups:         # Table 3b
        for m in ("lflstm", "tfn", "mult"):
            for c in ("clean", "AV30", "AV50"):
                R.append(RunCfg(model=m, train=TR[c], group="baselines"))
        R.append(RunCfg(dataset="mosi_unal", model="mult", train=TR["clean"], group="baselines"))
        R.append(RunCfg(dataset="mosi_unal", model="mult", train=TR["AV30"], group="baselines"))
    if "mosei" in groups:             # Table 5
        R.append(RunCfg(dataset="mosei", train=TR["clean"], group="mosei"))
        R.append(RunCfg(dataset="mosei", train=TR["AV30"], group="mosei"))
        R.append(RunCfg(dataset="mosei", train=TR["AV30"], mmr_mode="zero", group="mosei"))
    seen, out = set(), []
    for r in R:                        # same config requested by two groups -> run once
        if r.name not in seen:
            seen.add(r.name); out.append(r)
    return out


def build_model(cfg: RunCfg, dims) -> nn.Module:
    d_t, d_a, d_v = dims
    if cfg.model == "rmsa":
        return RMSANet(d_t, d_a, d_v, use_psg=cfg.use_psg, use_cmta=cfg.use_cmta, mmr_mode=cfg.mmr_mode)
    if cfg.model == "tfn":
        return TFN(d_t, d_a, d_v)
    if cfg.model == "lflstm":
        return LFLSTM(d_t, d_a, d_v)
    if cfg.model == "mult":
        return MulTCompact(d_t, d_a, d_v)
    raise ValueError(cfg.model)


# =====================================================================
# 7. Training (Section IV-B)
# =====================================================================
def _save_atomic(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)          # never leaves a half-written checkpoint (safe if Colab dies mid-save)


def train_one(cfg: RunCfg, data: dict, seed: int, epochs: int, bs: int, lr: float,
              ckpt_dir: Optional[str] = None, tag: str = "run", log=print):
    """Trains one (configuration, seed). If ckpt_dir is given, saves a checkpoint after EVERY epoch
    (model, optimiser, scheduler, best weights so far, learning curve and all RNG states) and resumes
    from it automatically, so an interrupted run continues from its last finished epoch with the
    same random stream as an uninterrupted run."""
    set_seed(seed)
    dims = tuple(data["train"][k].shape[2] for k in ("text", "audio", "vision"))
    model = build_model(cfg, dims).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr if cfg.model != "mult" else 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=0.0)     # Eq. (30)
    tr = Batcher(data["train"], bs, mask_seed=seed + 1000)
    va = Batcher(data["valid"], bs, mask_seed=seed + 2000)
    gen = np.random.RandomState(seed)
    best, best_state, best_ep, curve, start, spent = float("inf"), None, -1, [], 1, 0.0

    ck = os.path.join(ckpt_dir, f"{tag}_s{seed}.inprogress.pt") if ckpt_dir else None
    if ck and os.path.exists(ck):
        c = torch.load(ck, map_location=DEVICE, weights_only=False)
        if c.get("epochs") == epochs:
            model.load_state_dict(c["model"]); opt.load_state_dict(c["opt"]); sched.load_state_dict(c["sched"])
            best, best_ep, curve, spent = c["best"], c["best_ep"], c["curve"], c["spent"]
            best_state = {k: v.to(DEVICE) for k, v in c["best_state"].items()}
            gen.set_state(c["np_gen"]); torch.set_rng_state(c["torch_rng"])
            if torch.cuda.is_available() and c.get("cuda_rng") is not None:
                torch.cuda.set_rng_state_all(c["cuda_rng"])
            start = c["epoch"] + 1
            log(f"    resuming {tag} seed={seed} from epoch {start}/{epochs}")

    t0 = time.time()
    for ep in range(start, epochs + 1):
        model.train()
        for b in tr.iterate(cfg.train, train=True, gen=gen):
            out = model(b)
            loss, _ = rmsa_loss(out, b["y"], lam3=cfg.lam3)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)                       # Eq. (31)
            opt.step()
        sched.step()
        v = evaluate(model, va, cfg.train)            # model selection under the training condition
        curve.append((ep, v["mae"], v["acc2"]))
        if v["mae"] < best:
            best, best_ep = v["mae"], ep
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        if ck:
            _save_atomic(dict(epoch=ep, epochs=epochs, model=model.state_dict(), opt=opt.state_dict(),
                              sched=sched.state_dict(), best=best, best_ep=best_ep, curve=curve,
                              best_state={k: t.cpu() for k, t in best_state.items()},
                              spent=spent + time.time() - t0, np_gen=gen.get_state(), torch_rng=torch.get_rng_state(),
                              cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None), ck)
    model.load_state_dict(best_state)
    n_params = sum(p.numel() for p in model.parameters())
    return model, dict(best_epoch=best_ep, val_mae=best, train_sec=spent + time.time() - t0, n_params=n_params, curve=curve)


def load_trained(cfg: RunCfg, data: dict, seed: int, ckpt_dir: str) -> nn.Module:
    """Reload the best weights of a finished run (for new analyses without retraining)."""
    dims = tuple(data["train"][k].shape[2] for k in ("text", "audio", "vision"))
    model = build_model(cfg, dims).to(DEVICE)
    tag = cfg.name.replace("|", "_")
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, f"{tag}_s{seed}.best.pt"), map_location=DEVICE))
    return model


# =====================================================================
# 8. Analyses for R1.6 / R2.8 (PSG) and figure exports
# =====================================================================
@torch.no_grad()
def psg_sign_flip_check(model: RMSANet, batch) -> float:
    """Negating the last layer of the polarity estimator maps p -> -p; returns the max absolute change
    in the regression output (expected: 0 up to float error). Demonstrates R1.6's point."""
    if not model.use_psg:
        return float("nan")
    model.eval()
    y0 = model(batch)["y_reg"]
    model.psg.p2.weight.mul_(-1); model.psg.p2.bias.mul_(-1)
    y1 = model(batch)["y_reg"]
    model.psg.p2.weight.mul_(-1); model.psg.p2.bias.mul_(-1)
    return float((y0 - y1).abs().max())


@torch.no_grad()
def export_examples(model: RMSANet, data: dict, path: str, n_attn=8):
    """For the whole test set: PSG token weights (alpha, |p|) with words if raw_text exists (for the
    lexicon correlation and qualitative examples, R1.6/R2.8). For the first n_attn utterances: CMTA
    attention matrices (Figure 2, R2.9)."""
    model.eval()
    raw = data["test"].get("raw_text")
    tids = data["test"].get("text_ids")
    ex, attn = [], None
    for b in Batcher(data["test"], 64, mask_seed=0).iterate(Degrade("clean"), train=False):
        out = model(b)
        if attn is None:
            k = min(n_attn, len(b["y"]))
            attn = dict(A_ta=out["A_ta"][:k].cpu().numpy() if out["A_ta"] is not None else np.zeros(0),
                        A_tv=out["A_tv"][:k].cpu().numpy() if out["A_tv"] is not None else np.zeros(0),
                        tl=b["tl"][:k].cpu().numpy(), al=b["al"][:k].cpu().numpy(), vl=b["vl"][:k].cpu().numpy(),
                        index=b["idx"][:k].cpu().numpy())
        for i in range(len(b["y"])):
            idx = int(b["idx"][i]); T = int(b["tl"][i])
            ex.append(dict(index=idx, label=float(b["y"][i]), pred=float(out["y_reg"][i]),
                           words=str(raw[idx]).split() if raw is not None else None,
                           token_ids=tids[idx, :T].tolist() if tids is not None else None,
                           alpha=out["alpha"][i, :T].cpu().tolist() if out["alpha"] is not None else None,
                           abs_p=out["p"][i, :T].abs().cpu().tolist() if out["p"] is not None else None))
    np.savez_compressed(path + "_attn.npz", **attn)
    with open(path + "_psg.json", "w") as f:
        json.dump(ex, f)


def constant_predictor(data: dict) -> Dict[str, float]:
    """R2.3: MAE of always predicting the mean training label (and of predicting 0)."""
    y, mu = data["test"]["y"], float(data["train"]["y"].mean())
    return dict(const_mean_mae=float(np.abs(y - mu).mean()), const_zero_mae=float(np.abs(y).mean()),
                train_label_mean=mu)


# =====================================================================
# 9. Runner
# =====================================================================
def append_rows(csv_path: str, rows: List[dict]):
    import csv
    new = not os.path.exists(csv_path)
    keys = list(rows[0].keys())
    if not new:
        with open(csv_path) as f:
            header = next(csv.reader(f))
        keys = header + [k for k in keys if k not in header]
        if keys != header:      # header grew: rewrite file
            import pandas as pd
            df = pd.read_csv(csv_path)
            df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
            df.to_csv(csv_path, index=False); return
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


def done_runs(csv_path: str) -> set:
    if not os.path.exists(csv_path):
        return set()
    import pandas as pd
    df = pd.read_csv(csv_path)
    return set(zip(df["run"], df["seed"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mosi_aligned"); ap.add_argument("--mosi_unaligned"); ap.add_argument("--mosei")
    ap.add_argument("--groups", nargs="+", default=["main", "ablation", "mmr_analysis", "unaligned", "baselines"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[13, 42, 7, 17, 99])
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_standardize", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="synthetic data, 2 epochs, 2 seeds")
    ap.add_argument("--analyze_only", action="store_true", help="only rebuild tables/figures from per_run.csv")
    ap.add_argument("--no_keep_best", action="store_true", help="do not keep best weights of finished runs (saves Drive space)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.analyze_only:
        import analyze; analyze.run(a.out); return
    csv_path = os.path.join(a.out, "per_run.csv")
    ckpt_dir = os.path.join(a.out, "checkpoints"); os.makedirs(ckpt_dir, exist_ok=True)

    data = {}
    if a.smoke:
        data["mosi"] = synthetic_dataset(seed=0)
        data["mosi_unal"] = synthetic_dataset(T=(20, 120, 80), seed=1)
        data["mosei"] = synthetic_dataset(seed=2)
        a.epochs, a.seeds = 2, a.seeds[:2]
    else:
        for key, p in [("mosi", a.mosi_aligned), ("mosi_unal", a.mosi_unaligned), ("mosei", a.mosei)]:
            if p:
                data[key] = load_split_file(p)
    for k, d in data.items():
        if not a.no_standardize:
            standardize(d)
    info = {k: dict(stats=d.get("_stats", {}), **constant_predictor(d)) for k, d in data.items()}
    with open(os.path.join(a.out, "data_info.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(json.dumps(info, indent=1))

    runs = [r for r in registry(a.groups) if r.dataset in data]
    skipped = [r.name for r in registry(a.groups) if r.dataset not in data]
    if skipped:
        print(f"Skipping {len(skipped)} runs whose data set was not given: {sorted(set(s.split('|')[0] for s in skipped))}")
    done = done_runs(csv_path)
    todo = [(r, s) for s in a.seeds for r in runs if (r.name, s) not in done]
    print(f"{len(runs)} configurations x {len(a.seeds)} seeds; {len(todo)} runs to do (resume-safe).")

    for i, (cfg, seed) in enumerate(todo, 1):
        d = data[cfg.dataset]
        tag = cfg.name.replace("|", "_")
        model, tinfo = train_one(cfg, d, seed, a.epochs, a.bs, a.lr, ckpt_dir=ckpt_dir, tag=tag)
        te = Batcher(d["test"], 64, mask_seed=seed + 3000)
        rows = []
        for deg in TESTS:
            m = evaluate(model, te, deg, extras=(cfg.model == "rmsa"))
            rows.append(dict(run=cfg.name, group=cfg.group, dataset=cfg.dataset, model=cfg.model,
                             psg=int(cfg.use_psg), cmta=int(cfg.use_cmta), mmr_mode=cfg.mmr_mode, lam3=cfg.lam3,
                             train_pa=cfg.train.pa, train_pv=cfg.train.pv, seed=seed, test=deg.name,
                             best_epoch=tinfo["best_epoch"], val_mae=tinfo["val_mae"], n_params=tinfo["n_params"],
                             train_sec=round(tinfo["train_sec"], 1), epochs=a.epochs, **m))
        if cfg.model == "rmsa" and cfg.use_psg:
            b = next(Batcher(d["test"], 64, mask_seed=0).iterate(Degrade("clean"), train=False))
            rows[0]["psg_signflip_maxdiff"] = psg_sign_flip_check(model, b)
        if not a.no_keep_best:
            _save_atomic({k: t.cpu() for k, t in model.state_dict().items()}, os.path.join(ckpt_dir, f"{tag}_s{seed}.best.pt"))
        append_rows(csv_path, rows)                    # the run counts as done only after this line
        ip = os.path.join(ckpt_dir, f"{tag}_s{seed}.inprogress.pt")
        if os.path.exists(ip):
            os.remove(ip)
        with open(os.path.join(a.out, f"curve_{tag}_s{seed}.json"), "w") as f:
            json.dump(tinfo["curve"], f)
        if cfg.model == "rmsa" and cfg.use_psg and cfg.use_cmta and seed == a.seeds[0]:
            export_examples(model, d, os.path.join(a.out, f"examples_{tag}_s{seed}"))
        clean = rows[0]
        print(f"[{i}/{len(todo)}] {cfg.name} seed={seed} best_ep={tinfo['best_epoch']} "
              f"clean MAE={clean['mae']:.3f} Acc2={clean['acc2']:.2f} | {tinfo['train_sec']:.0f}s")

    # Build every table, statistic and figure of the paper from per_run.csv (also works on partial results)
    import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import analyze
    analyze.run(a.out)


if __name__ == "__main__":
    main()
