#!/usr/bin/env python3
"""
Figure 1 (>=300 DPI): composite of two benchmark-4 figures.

Panel A (top):    "Full EC Prediction Accuracy — All Models Compared"
                  re-rendered from plot_all_permutations.py (Plot 5) at 300 DPI.
Panel B (bottom): "MLP Mean Token - EC Prediction F1 Score by Layer"
                  re-rendered from plot_f1_and_confusion.py (Figure 1) at 300 DPI.

Data (finalized_codebase/models/benchmark_4/):
  xgboost_per_layer, xgboost_mean_per_layer, mlp_last_per_layer,
  mlp_mean_per_layer  -> results_summary.json (per-layer accuracies / F1s)
  error_bar_metrics.json -> Short/Medium/Long group ranges (A) and group F1
  SDs (B).

Output: <repo>/figures/benchmark_4/fig1_full_ec_and_f1_by_layer.png  (dpi=300)
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# Paths (script lives at <repo>/notebooks/stage_6_plots/)
# ============================================================
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
ROOT = os.path.abspath(os.path.join(REPO, ".."))
BENCH4_MODELS_DIR = os.path.join(ROOT, "finalized_codebase", "models", "benchmark_4")
OUTPUT_DIR = os.path.join(REPO, "figures", "benchmark_4")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LAYERS = list(range(32))
level_colors = {"1": "#2196F3", "2": "#4CAF50", "3": "#FF9800", "4": "#E91E63"}

CONFIGS = {
    "XGBoost Last Token": "xgboost_per_layer",
    "XGBoost Mean Token": "xgboost_mean_per_layer",
    "MLP Last Token":     "mlp_last_per_layer",
    "MLP Mean Token":     "mlp_mean_per_layer",
}
all_data = {}
for name, sub in CONFIGS.items():
    p = os.path.join(BENCH4_MODELS_DIR, sub, "results_summary.json")
    with open(p) as f:
        all_data[name] = json.load(f)
    print(f"Loaded: {name} ({len(all_data[name])} layers)")

with open(os.path.join(BENCH4_MODELS_DIR, "error_bar_metrics.json")) as f:
    eb_data = json.load(f)
print(f"Loaded error-bar metrics ({len(eb_data)} configs)")


def group_range(config, layer, level):
    """(min, max) of Short/Medium/Long group accuracy at one layer/level."""
    try:
        g = eb_data[config][str(layer)][f"level_{level}"]["groups"]
        vals = [g[k]["acc"] for k in ("Short", "Medium", "Long") if k in g]
        return (min(vals), max(vals)) if vals else None
    except (KeyError, TypeError):
        return None


CONFIG_B = "MLP Mean Token"

def group_f1_sd(layer, level):
    """Sample SD of Short/Medium/Long group macro F1 at one layer/level."""
    try:
        g = eb_data[CONFIG_B][str(layer)][f"level_{level}"]["groups"]
        vals = [g[k]["f1m"] for k in ("Short", "Medium", "Long") if k in g]
        return float(np.std(vals, ddof=1)) if len(vals) >= 2 else 0.0
    except (KeyError, TypeError):
        return 0.0


# ============================================================
# Composite figure: panel A on top, panel B below
# ============================================================
fig, (axA, axB) = plt.subplots(
    2, 1, figsize=(16, 14),
    gridspec_kw={"height_ratios": [7, 6], "hspace": 0.42},
)
x = np.arange(32)

# ---------------- Panel A ----------------
model_colors = {
    "XGBoost Last Token": "#2196F3",
    "XGBoost Mean Token": "#00BCD4",
    "MLP Last Token":     "#E91E63",
    "MLP Mean Token":     "#FF9800",
}
markers = {
    "XGBoost Last Token": "o",
    "XGBoost Mean Token": "s",
    "MLP Last Token":     "^",
    "MLP Mean Token":     "D",
}
for name, results in all_data.items():
    vals, los, his = [], [], []
    for l in LAYERS:
        r = results.get(str(l), {})
        vals.append(r.get("level_4_accuracy", 0.0))
        rng = group_range(name, l, 4)
        if rng is None:
            los.append(vals[-1]); his.append(vals[-1])
        else:
            los.append(rng[0]); his.append(rng[1])
    color = model_colors.get(name, "gray")
    axA.plot(x, vals, label=name, color=color,
             marker=markers.get(name, "o"), linewidth=2, markersize=5, alpha=0.85)
    axA.fill_between(x, los, his, alpha=0.15, color=color)

axA.set_xticks(x)
axA.set_xticklabels([str(l) for l in LAYERS])
axA.set_xlabel("Evo2 Layer", fontsize=12)
axA.set_ylabel("Full EC Level 4 Accuracy", fontsize=12)
axA.set_ylim(0, None)
axA.set_title("Full EC Prediction Accuracy — All Models Compared", fontsize=14, fontweight="bold")
axA.legend(fontsize=11)
axA.grid(axis="y", alpha=0.3)
axA.text(-0.035, 1.02, "A", transform=axA.transAxes, fontsize=20, fontweight="bold", va="bottom")

# ---------------- Panel B ----------------
results = all_data[CONFIG_B]
width = 0.2
for i, (level, color) in enumerate(level_colors.items()):
    vals, errs = [], []
    for l in LAYERS:
        r = results.get(str(l), {})
        vals.append(r.get(f"level_{level}_f1_macro", 0.0))
        errs.append(group_f1_sd(l, level))
    axB.bar(x + i * width, vals, width, label=f"Level {level}", color=color, alpha=0.85,
            yerr=errs, ecolor="black", capsize=1.2)

axB.set_xticks(x + 1.5 * width)
axB.set_xticklabels([str(l) for l in LAYERS])
axB.set_xlabel("Evo2 Layer")
axB.set_ylabel("F1 Score (macro)")
axB.set_ylim(0, 1.0)
axB.set_title("MLP Mean Token - EC Prediction F1 Score by Layer", fontsize=14, fontweight="bold")
axB.legend()
axB.grid(axis="y", alpha=0.3)
axB.text(-0.035, 1.02, "B", transform=axB.transAxes, fontsize=20, fontweight="bold", va="bottom")

out_path = os.path.join(OUTPUT_DIR, "fig1_full_ec_and_f1_by_layer.png")
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")
