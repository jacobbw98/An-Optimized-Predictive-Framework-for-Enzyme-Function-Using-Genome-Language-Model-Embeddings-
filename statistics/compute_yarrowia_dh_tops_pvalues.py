#!/usr/bin/env python3
"""P-tests for the benchmark-2 DH-MLP on the Yarrowia dataset.

(a) simple top-1 full-EC prediction vs the true full-EC: binomial test
    against chance 1/n_cls (n_cls = number of level-4 classes).
(b) simple top-3 full-EC prediction vs the true full-EC: binomial test
    against chance 3/n_cls.

Uses the saved top-3 predictions produced by
notebooks/stage_3_prediction_eval/predict_yarrowia_ec_deep.py
(publish_codebase/data/predictions_and_results/yarrowia_evo2_multilayer_embeddings_predicted.parquet),
i.e. the benchmark-2 DeepHierarchicalEC_MLP read off its level-4 head
directly ("simple" readout, no hierarchical constraint), same convention
as statistics/compute_benchmark4_pvalues.py.

Annotation subset: rows with EC_Numbers != 'UP' (UniProt placeholder for
uncharacterized function), the convention used by predict_yarrowia_ec_deep.py.
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

BASE = Path("/home/jacobbw/roell_koastore/Code2/Embeddings_Extraction")
PUB = BASE / "publish_codebase"
FIN = BASE / "finalized_codebase"
sys.path.insert(0, str(FIN / "notebooks"))
from definitions import ec_at_level  # noqa: E402

PRED_PQ = (PUB / "data" / "predictions_and_results"
           / "yarrowia_evo2_multilayer_embeddings_predicted.parquet")
ENC_PKL = FIN / "models" / "benchmark_2" / "deep_hierarchical_mlp_results" \
    / "label_encoders.pkl"


def main():
    t0 = time.time()
    df = pd.read_parquet(
        PRED_PQ,
        columns=["EC_Numbers", "Predicted_EC_L4_top1", "Predicted_EC_L4_top2",
                 "Predicted_EC_L4_top3"])
    df = df[df["EC_Numbers"] != "UP"].reset_index(drop=True)
    n = len(df)
    true4 = np.array([ec_at_level(e, 4) for e in df["EC_Numbers"]])
    h1 = true4 == df["Predicted_EC_L4_top1"].to_numpy()
    h3 = (h1 | (true4 == df["Predicted_EC_L4_top2"].to_numpy())
          | (true4 == df["Predicted_EC_L4_top3"].to_numpy()))

    enc = pickle.load(open(ENC_PKL, "rb"))
    n_cls = len(enc[4].classes_)

    b1 = binomtest(int(h1.sum()), n, 1 / n_cls, alternative="greater")
    b3 = binomtest(int(h3.sum()), n, 3 / n_cls, alternative="greater")

    out = {
        "dataset": ("Yarrowia, EC_Numbers != 'UP' "
                    "(predict_yarrowia_ec_deep.py convention)"),
        "model": ("benchmark-2 DH-MLP (DeepHierarchicalEC_MLP, "
                  "level-4 head, simple readout)"),
        "n": n,
        "n_classes_l4": n_cls,
        "top1": {
            "n_hit": int(h1.sum()), "acc": float(h1.mean()),
            "p0": 1 / n_cls, "p": float(b1.pvalue),
        },
        "top3": {
            "n_hit": int(h3.sum()), "acc": float(h3.mean()),
            "p0": 3 / n_cls, "p": float(b3.pvalue),
        },
    }
    print(json.dumps(out, indent=1))
    outp = Path(__file__).resolve().parent / "pvalues_yarrowia_dh_tops.json"
    outp.write_text(json.dumps(out, indent=1))
    print(f"saved {outp}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
