#!/usr/bin/env python3
"""Paired significance test: benchmark-2 DH-MLP vs flat MLP on benchmark-2 valid.

For every valid AC, both models are run with the "simple" level-4 readout
(argmax of the full-EC head, no hierarchical constraint) and the per-AC
full-EC correctness is compared with a paired t-test (scipy ttest_rel),
same style as statistics/compute_benchmark4_pvalues.py.

- flat MLP: models/benchmark_2/mlp_results/mlp_layer_18.pt
            (EC_MLP, 4096-dim layer-18 input)
- DH-MLP:   models/benchmark_2/deep_hierarchical_mlp_results/model.pt
            (DeepHierarchicalEC_MLP, 16384-dim input = concat of the
            4096-dim layer embeddings [13, 17, 18, 20], as in
            notebooks/stage_3_prediction_eval/predict_yarrowia_ec_deep.py)

Embeddings: data/benchmark_2/valid_embedding_batches{,_layer_13,_layer_17,
_layer_20} (pkl dicts keyed by AC).
True labels: data/benchmark_2/Valid.parquet, first EC per AC,
ec_at_level(EC, 4) (full EC).
Also reported: per-model binomial test of full-EC top-1 vs chance 1/n_cls.
"""
import gc
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from scipy.stats import binomtest, ttest_rel

BASE = Path("/home/jacobbw/roell_koastore/Code2/Embeddings_Extraction"
            "/finalized_codebase")
sys.path.insert(0, str(BASE / "notebooks"))
from definitions import (DeepHierarchicalEC_MLP, EC_MLP, ec_at_level)  # noqa: E402

VALID = BASE / "data" / "benchmark_2"
LAYER_DIRS = {
    18: VALID / "valid_embedding_batches",
    13: VALID / "valid_embedding_batches_layer_13",
    17: VALID / "valid_embedding_batches_layer_17",
    20: VALID / "valid_embedding_batches_layer_20",
}
BATCH = 1024


def load_layer(d):
    out = {}
    t0 = time.time()
    for f in sorted(d.glob("batch_*.pkl")):
        blob = pickle.load(open(f, "rb"))
        if not isinstance(blob, dict):
            continue
        for ac, val in blob.items():
            if isinstance(val, dict):  # {"blocks.L": [...]} form
                val = next(iter(val.values()))
            out[ac] = np.asarray(val, dtype=np.float32)
        del blob
        gc.collect()
    print(f"layer {d.name}: {len(out)} ACs "
          f"({time.time() - t0:.0f}s)", flush=True)
    return out


def argmax_l4(model, embs, acs, is_dh, enc4):
    hits = np.zeros(len(acs), dtype=bool)
    with torch.no_grad():
        for i in range(0, len(acs), BATCH):
            acs_b = acs[i:i + BATCH]
            X = torch.from_numpy(
                np.stack([embs[ac] for ac in acs_b]))
            logits = model(X)
            if is_dh:
                logits = logits[-1]
            pred = enc4.inverse_transform(
                np.asarray(logits.argmax(dim=1).cpu()))
            for j, ac in enumerate(acs_b):
                hits[i + j] = pred[j] == true4[ac]
            del X, logits, pred
        gc.collect()
    return hits


true4 = {}


def main():
    t0 = time.time()

    print("Loading true labels (first EC per AC)...", flush=True)
    tbl = pq.read_table(VALID / "Valid.parquet", columns=["AC", "EC"])
    for ac, ec in zip(tbl.column("AC").to_pylist(),
                      tbl.column("EC").to_pylist()):
        if ac not in true4:
            true4[ac] = ec_at_level(ec, 4)
    print(f"true labels: {len(true4)} ACs "
          f"({time.time() - t0:.0f}s)", flush=True)

    enc_f = pickle.load(open(
        BASE / "models/benchmark_2/mlp_results/label_encoder.pkl", "rb"))
    enc_d = pickle.load(open(
        BASE / "models/benchmark_2/deep_hierarchical_mlp_results"
        "/label_encoders.pkl", "rb"))
    n_cls = {
        "flat": len(enc_f.classes_),
        "dh": len(enc_d[4].classes_),
    }
    assert n_cls["flat"] == n_cls["dh"], n_cls

    embs = {}
    for lv in (18, 13, 17, 20):
        embs[lv] = load_layer(LAYER_DIRS[lv])

    acs = sorted(set(true4)
                 & set(embs[18]) & set(embs[13])
                 & set(embs[17]) & set(embs[20]))
    print(f"paired ACs (all 4 layers + label): {len(acs)} "
          f"({time.time() - t0:.0f}s)", flush=True)

    m_f = EC_MLP(4096, n_cls["flat"]).eval()
    m_f.load_state_dict(
        torch.load(BASE / "models/benchmark_2/mlp_results"
                   "/mlp_layer_18.pt",
                   map_location="cpu", weights_only=True))

    m_d = DeepHierarchicalEC_MLP(
        input_dim=16384,
        n_ec1=len(enc_d[1].classes_), n_ec2=len(enc_d[2].classes_),
        n_ec3=len(enc_d[3].classes_), n_ec4=n_cls["dh"],
        dropout=0.0).eval()
    sd = torch.load(BASE / "models/benchmark_2/deep_hierarchical_mlp_results"
                    "/model.pt", map_location="cpu", weights_only=True)
    m_d.load_state_dict({k.replace("_orig_mod.", ""): v
                         for k, v in sd.items()})
    del sd
    gc.collect()

    print("Running flat MLP...", flush=True)
    corr_f = argmax_l4(m_f, embs[18], np.array(acs), False, enc_f)
    acc_f = float(corr_f.mean())
    print(f"  flat top-1 full-EC acc = {acc_f:.6f} ({time.time() - t0:.0f}s)",
          flush=True)
    del m_f
    gc.collect()

    print("Running DH-MLP...", flush=True)
    dh_embs = {ac: np.concatenate(
        [embs[13][ac], embs[17][ac], embs[18][ac], embs[20][ac]])
        for ac in acs}
    for lv in (13, 17, 18, 20):
        del embs[lv]
    gc.collect()
    corr_d = argmax_l4(m_d, dh_embs, np.array(acs), True, enc_d[4])
    acc_d = float(corr_d.mean())
    print(f"  DH-MLP top-1 full-EC acc = {acc_d:.6f} ({time.time() - t0:.0f}s)",
          flush=True)

    t, p = ttest_rel(corr_d.astype(np.float64), corr_f.astype(np.float64))
    b_d = binomtest(int(corr_d.sum()), len(acs), 1 / n_cls["dh"],
                    alternative="greater")
    b_f = binomtest(int(corr_f.sum()), len(acs), 1 / n_cls["flat"],
                    alternative="greater")
    out = {
        "dataset": ("benchmark-2 valid, per-AC (first EC per AC), "
                    "all ACs present in the 4 embedding layers"),
        "models": {
            "dh_mlp": "benchmark-2 DeepHierarchicalEC_MLP, level-4 head, "
                      "simple readout",
            "flat_mlp": "benchmark-2 flat EC_MLP (layer 18, 4096-dim)",
        },
        "n_pairs": len(acs),
        "n_classes": n_cls["dh"],
        "acc_dh_mlp": acc_d,
        "acc_flat_mlp": acc_f,
        "paired_ttest_rel": {
            "t": float(t), "p": float(p),
            "note": "per-AC full-EC correctness, DH-MLP vs flat MLP",
        },
        "dh_mlp_vs_chance": {"p0": 1 / n_cls["dh"], "p": float(b_d.pvalue)},
        "flat_mlp_vs_chance": {"p0": 1 / n_cls["flat"],
                               "p": float(b_f.pvalue)},
    }
    print(json.dumps(out, indent=1))
    outp = Path(__file__).resolve().parent / "pvalues_b2_dh_vs_flat.json"
    outp.write_text(json.dumps(out, indent=1))
    print(f"saved {outp}  (total {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
