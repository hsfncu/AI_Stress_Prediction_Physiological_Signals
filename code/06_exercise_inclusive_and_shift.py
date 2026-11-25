"""P1 延伸：訓練是否含運動視窗 + 分布差異量化。
A. 留一受試者：rest-only 訓練 vs exercise-inclusive 訓練（運動標成非壓力），報告每人 壓力召回 / 靜息誤判 / 運動誤判
B. 分布差異：靜息、壓力、運動三種狀態在各特徵上的標準化平均差（以受試者平均為單位）與 Wasserstein 距離
用法：python 06_exercise_inclusive_and_shift.py <windows.csv> <輸出資料夾>
"""
import sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import wasserstein_distance, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

Wp, OUT = sys.argv[1:3]
df = pd.read_csv(Wp); df["hrv_missing"] = df.rmssd.isna().astype(float)
FULL = ["hr_mean", "hr_std", "hr_slope", "rmssd", "sdnn", "meannn", "acc_mean", "acc_std", "acc_p95", "hrv_missing"]
NOACC = ["hr_mean", "hr_std", "hr_slope", "rmssd", "sdnn", "meannn", "hrv_missing"]
res = {}

def prep(tr, te, cols):
    med = tr[cols].median(); a, b = tr[cols].fillna(med), te[cols].fillna(med); mu, sd = a.mean(), a.std().replace(0, 1)
    return ((a - mu) / sd).values, ((b - mu) / sd).values

subs = sorted(set(df[df.label == "exercise"].subject) & set(df[df.label == "stress"].subject) & set(df[df.label == "rest"].subject))
rows = []
for model in ["LogReg", "RandomForest"]:
    for cname, cols in [("with ACC", FULL), ("without ACC", NOACC)]:
        for train_set in ["rest-only", "exercise-inclusive"]:
            for s in subs:
                tr = df[df.subject != s]; te = df[df.subject == s]
                tr = tr[tr.label.isin(["rest", "stress"])] if train_set == "rest-only" else tr
                ytr = (tr.label == "stress").astype(int).values
                Xa, Xb = prep(tr, te, cols)
                clf = LogisticRegression(max_iter=2000, class_weight="balanced") if model == "LogReg" else RandomForestClassifier(200, min_samples_leaf=3, class_weight="balanced", random_state=0, n_jobs=4)
                pr = clf.fit(Xa, ytr).predict(Xb); lab = te.label.values
                rows.append(dict(model=model, features=cname, train=train_set, subject=s,
                                 stress_recall=float((pr[lab == "stress"] == 1).mean()), rest_fp=float((pr[lab == "rest"] == 1).mean()),
                                 exercise_fp=float((pr[lab == "exercise"] == 1).mean())))
r = pd.DataFrame(rows); r.to_csv(f"{OUT}/p1b_exercise_inclusive.csv", index=False)
summ = {}
for (model, f, t), g in r.groupby(["model", "features", "train"]):
    summ[f"{model}|{f}|{t}"] = {c: [round(float(g[c].median()), 3), round(float(g[c].quantile(.25)), 3), round(float(g[c].quantile(.75)), 3)] for c in ["stress_recall", "rest_fp", "exercise_fp"]}
    print(f"{model:12s} {f:12s} {t:19s} n={len(g)} | stress recall {summ[f'{model}|{f}|{t}']['stress_recall']} | rest FP {summ[f'{model}|{f}|{t}']['rest_fp']} | exercise FP {summ[f'{model}|{f}|{t}']['exercise_fp']}")
res["exercise_inclusive"] = summ
# 配對檢定：exercise-inclusive 相對 rest-only 的差（每人）
for (model, f), g in r.groupby(["model", "features"]):
    a = g[g.train == "rest-only"].set_index("subject"); b = g[g.train == "exercise-inclusive"].set_index("subject")
    d_ex = (b.exercise_fp - a.exercise_fp); d_rec = (b.stress_recall - a.stress_recall)
    res.setdefault("paired", {})[f"{model}|{f}"] = dict(d_exercise_fp_median=float(d_ex.median()), p_exercise=float(wilcoxon(b.exercise_fp, a.exercise_fp).pvalue),
                                                        d_stress_recall_median=float(d_rec.median()), p_recall=float(wilcoxon(b.stress_recall, a.stress_recall).pvalue))
    print("paired", model, f, json.dumps(res["paired"][f"{model}|{f}"]))

# ---------- B. 分布差異 ----------
feat = ["hr_mean", "rmssd", "acc_std", "acc_mean", "hrv_missing"]
cond_means = df.groupby(["subject", "label"])[feat].mean().reset_index()
smd = {}
for a, b in [("rest", "stress"), ("rest", "exercise"), ("stress", "exercise")]:
    for f in feat:
        x = cond_means[cond_means.label == a].set_index("subject")[f]; y = cond_means[cond_means.label == b].set_index("subject")[f]
        j = x.index.intersection(y.index); d = (y[j] - x[j]).dropna()
        smd[f"{a}->{b}|{f}"] = dict(paired_d=float(d.mean() / d.std()) if d.std() > 0 else np.nan, mean_diff=float(d.mean()), n=int(len(d)))
res["shift_paired_effect_size"] = smd
for k, v in smd.items(): print("shift", k, "paired d=%.2f meandiff=%.3f n=%d" % (v["paired_d"], v["mean_diff"], v["n"]))
w = {}
for f in ["hr_mean", "acc_std", "acc_mean"]:
    z = (df[f] - df[f].mean()) / df[f].std()
    for a, b in [("rest", "stress"), ("rest", "exercise"), ("stress", "exercise")]:
        w[f"{a}-{b}|{f}"] = float(wasserstein_distance(z[df.label == a].dropna(), z[df.label == b].dropna()))
res["wasserstein_standardized"] = w
print("wasserstein", json.dumps({k: round(v, 3) for k, v in w.items()}))
json.dump(res, open(f"{OUT}/p1b_results.json", "w"), indent=1, default=float)
