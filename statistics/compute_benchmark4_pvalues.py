#!/usr/bin/env python
"""Significance testing for Benchmark 4 model comparisons.

Reproduces the per-sequence level-4 evaluation of
notebooks/stage_5_benchmark4_training/evaluate_benchmark4_error_bars.py
(its masks, label encoders, ec_at_level(.,4) comparison, and sequence
length tertiles), then adds:

  * paired two-tailed Student's t-test (scipy.stats.ttest_rel) between
    two models' per-sequence level-4 correctness vectors, paired over
    the intersection of their per-column validity masks;
  * MLP-vs-random-guessing via exact binomial test
    (scipy.stats.binomtest, alternative="greater") of the correctness
    count against the uniform chance rate 1/n_cls.

DH-MLP (MLP Mean, L17, 0.7347) is compared against all other methods
(XGB Last L17, XGB Mean L19, MLP Last L17, RF Block/MLP/Post-Norm L24,
RF Combined L9+24+26); RF Combined is also compared against each
single-layer RF, and XGB Last vs XGB Mean.

The 62-71 GB Random Forest pickles are streamed: during unpickle,
DecisionTreeClassifier.__setstate__ is patched so that each fitted tree
accumulates its predict_proba(X) and is then dropped, bounding peak
memory to ~one tree + X (a plain load needs ~133 GB). Final class =
argmax of the summed per-tree proba, identical to
RandomForestClassifier.predict (mean-proba argmax).

Outputs (next to this script): pvalues_benchmark4.csv,
pvalues_benchmark4.json (models, groups, comparisons, binomial tests,
sanity check vs published accuracies).

Usage:
  python compute_benchmark4_pvalues.py --project-root /path/to/publish_codebase
"""
import argparse
import gc
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import warnings
from scipy import stats
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

DIM = 4096
LV = 4  # hierarchy level compared (level-4 accuracy)


def ec_at_level(ec_str, level):
    """Truncate an EC number to a hierarchy level (definitions.py)."""
    parts = str(ec_str).split(".")
    return ".".join(parts[:level])


class EC_MLP_Compact(torch.nn.Module):
    """Compact 3-layer MLP (notebooks/definitions.py) — exact copy."""

    def __init__(self, dim, nclass, drop=0.3):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, 1024), torch.nn.ReLU(), torch.nn.Dropout(drop),
            torch.nn.Linear(1024, 512), torch.nn.ReLU(), torch.nn.Dropout(drop),
            torch.nn.Linear(512, nclass),
        )

    def forward(self, x):
        return self.net(x)


def list_col_to_matrix(chunked, n):
    """LargeList<float64> column -> (float32 (n, DIM), valid_bool_mask)."""
    la = chunked.combine_chunks()
    valid = ~np.asarray(la.is_null(), dtype=bool)
    vals = np.asarray(la.values.to_numpy(), dtype=np.float32)
    mat = np.zeros((n, DIM), dtype=np.float32)
    if valid.any():
        mat[valid] = vals.reshape(int(valid.sum()), DIM)
    del vals
    return mat, valid


def load_col(pq_path, cols):
    """Load one or more embedding columns -> ({col: mat}, {col: mask})."""
    tab = pq.read_table(pq_path, columns=list(cols))
    mats, masks = {}, {}
    for c in cols:
        mats[c], masks[c] = list_col_to_matrix(tab.column(c), tab.num_rows)
    del tab
    return mats, masks


class ForestStreamPredictor:
    """Predict with a huge pickled RandomForest without loading it whole.

    Peak memory ~= one tree + X. The C pickle implementation restores
    estimator state via ``__setstate__`` (``load_build`` overrides are
    never called), so the hook is applied to
    DecisionTreeClassifier.__setstate__ for the duration of the load:
    each fitted tree accumulates its predict_proba(X) and is then
    dropped (the forest's unfitted ``estimator_`` template, which has
    no ``tree_``, is skipped). Final class = argmax of the summed
    per-tree proba, identical to RandomForestClassifier.predict
    (mean-proba argmax).
    """

    def __init__(self, X, verbose=True):
        self.X = X
        self.verbose = verbose
        self.proba_sum = None
        self.n_trees = 0

    def predict(self, path):
        me = self
        holder = next(k for k in DecisionTreeClassifier.__mro__
                      if "__setstate__" in vars(k))
        orig = vars(holder)["__setstate__"]

        def hooked(self, state):
            if "tree_" not in state:  # unfitted estimator_ template
                orig(self, state)
                return
            orig(self, state)
            p = self.predict_proba(me.X)
            if me.proba_sum is None:
                me.proba_sum = np.zeros(
                    (me.X.shape[0], p.shape[1]), dtype=np.float64)
            me.proba_sum += p
            me.n_trees += 1
            # drop the tree: the forest keeps only an empty shell
            self.tree_ = None
            self.classes_ = None
            del p
            if me.n_trees % 50 == 0:
                if me.verbose:
                    print(f"      {me.n_trees} trees, "
                          f"{time.strftime('%H:%M:%S')}", flush=True)
                gc.collect()

        DecisionTreeClassifier.__setstate__ = hooked
        try:
            with open(path, "rb") as f:
                pickle.load(f)
        finally:
            del DecisionTreeClassifier.__setstate__
        if me.verbose:
            print(f"      {me.n_trees} trees total", flush=True)
        return np.argmax(self.proba_sum, axis=1), self.n_trees


def mlp_argmax(model, X, bs=4096):
    model.eval()
    with torch.no_grad():
        out = [model(torch.from_numpy(X[i:i + bs])).argmax(1)
               for i in range(0, len(X), bs)]
    return torch.cat(out).numpy().astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True,
                    help="Populated tree with data/ and models/ "
                         "(e.g. .../publish_codebase)")
    args = ap.parse_args()
    root = Path(args.project_root).resolve()
    out_dir = Path(__file__).resolve().parent
    B4 = root / "data" / "benchmark_4"
    MD = root / "models" / "benchmark_4"
    ALLPQ = B4 / "Valid_all_layers_embeddings.parquet"
    RFPQ = B4 / "Valid_with_embeddings.parquet"
    t0 = time.time()

    # ============ labels, sequence length groups (published pipeline) ============
    df_seq = pq.read_table(B4 / "Valid.parquet",
                           columns=["AC", "Sequence"]).to_pandas()
    seq_lens = df_seq["Sequence"].str.len().to_numpy()
    ac_seq = df_seq["AC"].to_numpy()
    del df_seq
    n = len(seq_lens)
    order = np.argsort(seq_lens)
    t1, t2 = n // 3, 2 * n // 3
    grp = np.empty(n, dtype=np.int32)
    grp[order[:t1]] = 0
    grp[order[t1:t2]] = 1
    grp[order[t2:]] = 2
    srt = seq_lens[order]
    gnames = ["Short", "Medium", "Long"]
    ginfo = {
        "Short": {"n": t1, "len_range": [int(seq_lens.min()), int(srt[t1 - 1])]},
        "Medium": {"n": t2 - t1, "len_range": [int(srt[t1]), int(srt[t2 - 1])]},
        "Long": {"n": n - t2, "len_range": [int(srt[t2]), int(seq_lens.max())]},
    }
    print(f"Validation set: {n} sequences")
    for g in gnames:
        gi = ginfo[g]
        print(f"  {g}: n={gi['n']}, len {gi['len_range'][0]}-{gi['len_range'][1]}")

    ec_str = np.asarray(pq.read_table(ALLPQ, columns=["EC"])
                        .column("EC").to_pylist())
    ac_emb = np.asarray(pq.read_table(ALLPQ, columns=["AC"])
                        .column("AC").to_pylist())
    if not np.array_equal(ac_seq, ac_emb):
        raise SystemExit("Row order mismatch between Valid.parquet and "
                         "Valid_all_layers_embeddings.parquet")
    true_lv4 = np.array([ec_at_level(s, LV) for s in ec_str])

    train_ec = np.asarray(pq.read_table(
        B4 / "Train_all_layers_embeddings.parquet", columns=["EC"])
        .column("EC").to_pylist())
    le = LabelEncoder().fit(np.concatenate([train_ec, ec_str]))
    n_cls = len(le.classes_)
    print(f"Label classes (train+valid full EC): {n_cls}")
    del train_ec
    gc.collect()

    models = {}  # name -> dict(mask, corr)

    def add(name, pred_idx, enc, mask):
        pred_lv4 = np.array([ec_at_level(s, LV)
                             for s in enc.inverse_transform(pred_idx)])
        corr = (true_lv4 == pred_lv4)
        acc = corr[mask].mean()
        models[name] = dict(mask=mask, corr=corr)
        print(f"  {name:28s} level-4 acc = {acc:.4f}  (n={int(mask.sum())})",
              flush=True)

    # ============ XGBoost (best layers L17 last, L19 mean) ============
    print("Evaluating XGBoost models...", flush=True)
    mats, masks = load_col(ALLPQ, ["emb_blocks_17"])
    X, m17 = mats["emb_blocks_17"], masks["emb_blocks_17"]
    m = XGBClassifier()
    m.load_model(str(MD / "xgboost_per_layer" / "xgb_layer_17.json"))
    add("XGB Last (L17)", m.predict(X[m17]), le, m17)
    del X, mats, m
    gc.collect()

    mats, masks = load_col(ALLPQ, ["emb_mean_blocks_19"])
    X, m19 = mats["emb_mean_blocks_19"], masks["emb_mean_blocks_19"]
    m = XGBClassifier()
    m.load_model(str(MD / "xgboost_mean_per_layer" / "xgb_mean_layer_19.json"))
    add("XGB Mean (L19)", m.predict(X[m19]), le, m19)
    del X, mats, m
    gc.collect()

    # ============ MLP (best layer L17, last and mean) ============
    print("Evaluating MLP models (CPU)...", flush=True)
    mats, masks = load_col(ALLPQ, ["emb_blocks_17"])
    X, m17 = mats["emb_blocks_17"], masks["emb_blocks_17"]
    mdl = EC_MLP_Compact(DIM, n_cls)
    mdl.load_state_dict(torch.load(MD / "mlp_last_per_layer" /
                                   "mlp_last_layer_17.pt",
                                   map_location="cpu", weights_only=True))
    add("MLP Last (L17)", mlp_argmax(mdl, X[m17]), le, m17)
    del X, mats, mdl
    gc.collect()

    mats, masks = load_col(ALLPQ, ["emb_mean_blocks_17"])
    X, m17m = mats["emb_mean_blocks_17"], masks["emb_mean_blocks_17"]
    mdl = EC_MLP_Compact(DIM, n_cls)
    mdl.load_state_dict(torch.load(MD / "mlp_mean_per_layer" /
                                   "mlp_mean_layer_17.pt",
                                   map_location="cpu", weights_only=True))
    add("MLP Mean (L17)", mlp_argmax(mdl, X[m17m]), le, m17m)
    del X, mats, mdl
    gc.collect()

    # ============ Random Forests (streamed, best layer L24 + combined) ============
    print("Evaluating Random Forest models (streamed)...", flush=True)
    rf_le = pickle.load(open(MD / "rf_benchmark4" / "label_encoder.pkl", "rb"))
    rf_dir = MD / "rf_benchmark4"

    for col, mfile, name in [
        ("emb_blocks_24", "rf_layer_24_block.pkl", "RF Block (L24)"),
        ("emb_blocks_24_mlp", "rf_layer_24_mlp.pkl", "RF MLP (L24)"),
        ("emb_blocks_24_post_norm", "rf_layer_24_post_norm.pkl",
         "RF Post-Norm (L24)"),
    ]:
        mats, masks = load_col(RFPQ, [col])
        X, mk = mats[col], masks[col]
        print(f"  {name} (streaming {mfile})...", flush=True)
        pred, nt = ForestStreamPredictor(X[mk]).predict(rf_dir / mfile)
        add(name, pred, rf_le, mk)
        del X, mats, pred
        gc.collect()

    comb_cols = [f"emb_blocks_{L}{sfx}"
                 for L in (9, 24, 26)
                 for sfx in ("", "_mlp", "_post_norm")]
    print(f"  RF Combined ({len(comb_cols)} x {DIM} = "
          f"{len(comb_cols) * DIM} features) streaming...", flush=True)
    mats, masks = load_col(RFPQ, comb_cols)
    X = np.concatenate([mats[c] for c in comb_cols], axis=1)
    mk = masks[comb_cols[0]]
    del mats
    gc.collect()
    pred, nt = ForestStreamPredictor(X[mk]).predict(
        MD / "rf_combined_results" / "rf_combined_model.pkl")
    add("RF Combined (L9+24+26)", pred, rf_le, mk)
    del X, pred
    gc.collect()

    # ============ comparisons: paired two-tailed t-tests ============
    pairs = [
        ("MLP Mean (L17)", "XGB Last (L17)"),
        ("MLP Mean (L17)", "XGB Mean (L19)"),
        ("MLP Mean (L17)", "MLP Last (L17)"),
        ("MLP Mean (L17)", "RF Block (L24)"),
        ("MLP Mean (L17)", "RF MLP (L24)"),
        ("MLP Mean (L17)", "RF Post-Norm (L24)"),
        ("MLP Mean (L17)", "RF Combined (L9+24+26)"),
        ("RF Combined (L9+24+26)", "RF Block (L24)"),
        ("RF Combined (L9+24+26)", "RF MLP (L24)"),
        ("RF Combined (L9+24+26)", "RF Post-Norm (L24)"),
        ("XGB Last (L17)", "XGB Mean (L19)"),
    ]
    cmps = []
    for a, b in pairs:
        ma = models[a]["mask"] & models[b]["mask"]
        ca = models[a]["corr"][ma].astype(np.float64)
        cb = models[b]["corr"][ma].astype(np.float64)
        t, p = stats.ttest_rel(ca, cb)
        cmps.append(dict(name=f"{a} vs {b}",
                         acc_a=float(models[a]["corr"][ma].mean()),
                         acc_b=float(models[b]["corr"][ma].mean()),
                         n=int(ma.sum()), t=float(t), p=float(p)))

    # ============ MLP vs random guessing (exact binomial) ============
    guess = []
    for name in ("MLP Last (L17)", "MLP Mean (L17)"):
        mk = models[name]["mask"]
        k, nn = int(models[name]["corr"][mk].sum()), int(mk.sum())
        first = np.fromiter((ord(s[0]) for s in true_lv4[mk]),
                            dtype=np.int64)
        counts = np.bincount(first)
        most_common = float(counts.max()) / nn
        bt = stats.binomtest(k, nn, 1.0 / n_cls, alternative="greater")
        guess.append(dict(name=name, n=nn, correct=k,
                          acc=float(models[name]["corr"][mk].mean()),
                          n_cls=n_cls,
                          chance_uniform=1.0 / n_cls,
                          chance_most_common_class=most_common,
                          p_binom_greater=float(bt.pvalue)))

    # ============ per-model summary incl. length-group accuracies ============
    summary = []
    for name, d in models.items():
        mk, corr = d["mask"], d["corr"]
        byg = []
        for gi, g in enumerate(gnames):
            sel = mk & (grp == gi)
            byg.append(dict(group=g, acc=float(corr[sel].mean()),
                            n=int(sel.sum()),
                            std=float(np.std(corr[sel], ddof=1))))
        summary.append(dict(model=name, acc=float(corr[mk].mean()),
                            n=int(mk.sum()), by_length_group=byg))

    expected = {"XGB Last (L17)": 0.3897, "XGB Mean (L19)": 0.6208,
                "MLP Last (L17)": 0.6333, "MLP Mean (L17)": 0.7347,
                "RF Block (L24)": 0.1191, "RF MLP (L24)": 0.1007,
                "RF Post-Norm (L24)": 0.1162,
                "RF Combined (L9+24+26)": 0.1331}
    sanity = []
    for s in summary:
        exp = expected[s["model"]]
        sanity.append(dict(model=s["model"], acc=s["acc"], published=exp,
                           match=abs(s["acc"] - exp) < 0.0005))

    out = dict(
        n_sequences=n, n_classes=int(n_cls), level=LV,
        length_groups=ginfo,
        models=summary, comparisons=cmps, random_guessing=guess,
        sanity=sanity, wall_time_s=round(time.time() - t0, 1),
        method=("Paired two-tailed Student's t-test "
                "(scipy.stats.ttest_rel) on per-sequence level-4 correctness "
                f"vectors (correctness = ec_at_level(pred,{LV}) == "
                "ec_at_level(true,4)), paired over the intersection of each "
                "model's per-column validity mask. MLP vs random: exact "
                "binomial test, alternative='greater', p0 = 1/n_cls."),
    )
    (out_dir / "pvalues_benchmark4.json").write_text(json.dumps(out, indent=2))
    csv = ["comparison,acc_a,acc_b,n,t,p"]
    for c in cmps:
        csv.append(f'{c["name"]},{c["acc_a"]:.4f},{c["acc_b"]:.4f},'
                   f'{c["n"]},{c["t"]:.4f},{c["p"]:.4e}')
    (out_dir / "pvalues_benchmark4.csv").write_text("\n".join(csv) + "\n")

    # ============ console report ============
    print("\n=== Sanity (recomputed vs published accuracy) ===")
    for s in sanity:
        print(f"  {s['model']:28s} {s['acc']:.4f}  published {s['published']:.4f}"
              f"  {'OK' if s['match'] else 'MISMATCH!'}")
    print("\n=== Accuracy by length group (acc +/- std, ddof=1) ===")
    print(f"  {'group':7s}" + "".join(f"{s['model'][:12]:>14s}" for s in summary))
    for gi, g in enumerate(gnames):
        row = f"  {g:7s}"
        for s in summary:
            b = s["by_length_group"][gi]
            row += f"{b['acc']:.4f}{b['std']:+.3f}".rjust(14)
        print(row)
    print("\n=== Paired two-tailed t-tests ===")
    for c in cmps:
        print(f"  {c['name']:52s} {c['acc_a']:.4f} vs {c['acc_b']:.4f}"
              f"  n={c['n']}  t={c['t']:9.3f}  p={c['p']:.3e}")
    print("\n=== MLP vs random guessing (exact binomial, greater) ===")
    for g in guess:
        print(f"  {g['name']:15s} acc={g['acc']:.4f}  "
              f"chance_uniform=1/{g['n_cls']}={g['chance_uniform']:.6f}  "
              f"chance_most_common={g['chance_most_common_class']:.4f}  "
              f"p={g['p_binom_greater']:.3e}")
    print(f"\nDone in {time.time() - t0:.1f}s. Wrote "
          f"{out_dir / 'pvalues_benchmark4.csv'} and .json")


if __name__ == "__main__":
    sys.exit(main())
