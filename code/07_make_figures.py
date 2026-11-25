"""產生論文圖表（Fig 1-3）與需要引用的補充數字。
用法：python 07_make_figures.py <results資料夾> <輸出資料夾>
"""
import sys, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R, OUT = sys.argv[1:3]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GREY, GREEN = "#0072B2", "#E69F00", "#666666", "#009E73"

w = pd.read_csv(f"{R}/windows.csv"); w["hrv_missing"] = w.rmssd.isna().astype(float)
inv = pd.read_csv(f"{R}/inventory.csv")
info = {}
core = w[w.label.isin(["rest", "stress"])]
info["participants_stress_protocol"] = int(core.subject.nunique())
info["participants_by_stage"] = core.groupby("stage").subject.nunique().to_dict()
info["participants_with_exercise"] = int(w[w.label == "exercise"].subject.nunique())
info["participants_all_three"] = int(sum(1 for _, g in w.groupby("subject") if g.label.nunique() == 3))
info["windows"] = w.label.value_counts().to_dict()
info["windows_by_task"] = w.groupby(["label", "task"]).size().to_dict().__repr__()
info["windows_by_stage"] = w.groupby(["stage", "label"]).size().to_dict().__repr__()
print(json.dumps(info, indent=1, default=str))

# ---- Fig 1: 每人被判成壓力的比率（rest-only vs exercise-inclusive，邏輯迴歸 + 加速度）----
p = pd.read_csv(f"{R}/p1b_exercise_inclusive.csv")
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), sharey=True)
for ax, tr, ttl in zip(axes, ["rest-only", "exercise-inclusive"], ["(a) Trained on stress vs rest only", "(b) Exercise added as a non-stress class"]):
    g = p[(p.model == "LogReg") & (p.features == "with ACC") & (p.train == tr)]
    data = [g.stress_recall, g.rest_fp, g.exercise_fp]
    bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False, medianprops=dict(color="black"))
    for patch, c in zip(bp["boxes"], [GREEN, BLUE, ORANGE]): patch.set_facecolor(c); patch.set_alpha(0.35)
    rng = np.random.default_rng(1)
    for i, (d, c) in enumerate(zip(data, [GREEN, BLUE, ORANGE]), start=1):
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(d)), d, s=9, color=c, alpha=0.8, zorder=3)
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(["Stress\nwindows", "Rest\nwindows", "Exercise\nwindows"]); ax.set_title(ttl, fontsize=9)
axes[0].set_ylabel("Proportion classified as stress\n(per participant)")
plt.tight_layout(); plt.savefig(f"{OUT}/Fig1_exercise_false_stress.png", dpi=300); plt.close()

# ---- Fig 2: 三種狀態的分布（受試者平均）----
cm = w.groupby(["subject", "label"])[["hr_mean", "acc_std", "hrv_missing"]].mean().reset_index()
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8))
for ax, f, ylab in zip(axes, ["hr_mean", "acc_std", "hrv_missing"], ["Heart rate (bpm)", "Acceleration SD (g)", "Share of windows without PRV"]):
    data = [cm[cm.label == s][f].dropna() for s in ["rest", "stress", "exercise"]]
    bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False, medianprops=dict(color="black"))
    for patch, c in zip(bp["boxes"], [BLUE, GREEN, ORANGE]): patch.set_facecolor(c); patch.set_alpha(0.35)
    rng = np.random.default_rng(2)
    for i, (d, c) in enumerate(zip(data, [BLUE, GREEN, ORANGE]), start=1): ax.scatter(i + rng.uniform(-0.15, 0.15, len(d)), d, s=8, color=c, alpha=0.8, zorder=3)
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(["Rest", "Stress", "Exercise"]); ax.set_ylabel(ylab)
plt.tight_layout(); plt.savefig(f"{OUT}/Fig2_state_distributions.png", dpi=300); plt.close()

# ---- Fig 3: risk-coverage ----
ext = json.load(open(f"{R}/extended_results.json"))["D_selective"]
fig, ax = plt.subplots(figsize=(3.4, 2.8))
for name, c in [("LogReg", BLUE), ("RandomForest", ORANGE)]:
    cov = [x[0] * 100 for x in ext[name]]; ba = [x[1] for x in ext[name]]
    ax.plot(cov, ba, marker="o", color=c, label="Logistic regression" if name == "LogReg" else "Random forest")
ax.axhline(0.5, color=GREY, ls="--", lw=0.8); ax.set_xlabel("Coverage (% of windows classified)"); ax.set_ylabel("Balanced accuracy"); ax.invert_xaxis(); ax.legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/Fig3_risk_coverage.png", dpi=300); plt.close()
json.dump(info, open(f"{OUT}/figure_info.json", "w"), indent=1, default=str)
print("figures done")
