"""ChatGPT 建議後的優先分析（結果自行驗證）。
P1  壓力 vs 靜息模型（訓練不含運動視窗）套到運動視窗：每人誤判成壓力的比率
P2  受試者層級配對比較（邏輯迴歸 vs 隨機森林）
P3  用獨立任務重新檢驗「HR 反應幅度 vs 辨識表現」（避免數學耦合）
P4  自評壓力：只取壓力任務階段，混合效應模型（受試者隨機截距）
P5  對照：視窗層級隨機 10 折 vs 留一受試者（顯示切分方式造成的差距）
用法：python 05_priority_analyses.py <windows.csv> <loso_probs.csv> <phase_selfreport.csv> <輸出資料夾>
"""
import sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

Wp, Pp, Sp, OUT = sys.argv[1:5]
df = pd.read_csv(Wp); df["hrv_missing"] = df.rmssd.isna().astype(float)
lp = pd.read_csv(Pp)
FULL = ["hr_mean", "hr_std", "hr_slope", "rmssd", "sdnn", "meannn", "acc_mean", "acc_std", "acc_p95", "hrv_missing"]
NOACC = ["hr_mean", "hr_std", "hr_slope", "rmssd", "sdnn", "meannn", "hrv_missing"]
res = {}

def prep(tr, te, cols):
    med = tr[cols].median(); a, b = tr[cols].fillna(med), te[cols].fillna(med); mu, sd = a.mean(), a.std().replace(0, 1)
    return ((a - mu) / sd).values, ((b - mu) / sd).values

# ---------- P1 ----------
core = df[df.label.isin(["rest", "stress"])].reset_index(drop=True)
ex = df[df.label == "exercise"].reset_index(drop=True)
rows = []
for cols_name, cols in [("with ACC", FULL), ("without ACC", NOACC)]:
    for s in sorted(set(core.subject) & set(ex.subject)):
        tr = core[core.subject != s]; te_core = core[core.subject == s]; te_ex = ex[ex.subject == s]
        ytr = (tr.label == "stress").astype(int).values
        Xa, Xb = prep(tr, pd.concat([te_core, te_ex]), cols)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xa, ytr)
        pr = clf.predict(Xb); n1 = len(te_core)
        lab = te_core.label.values
        rows.append(dict(features=cols_name, subject=s, rest_fp=float((pr[:n1][lab == "rest"] == 1).mean()) if (lab == "rest").any() else np.nan,
                         stress_recall=float((pr[:n1][lab == "stress"] == 1).mean()) if (lab == "stress").any() else np.nan,
                         exercise_false_stress=float((pr[n1:] == 1).mean()), n_ex=len(te_ex)))
p1 = pd.DataFrame(rows); p1.to_csv(f"{OUT}/p1_exercise_false_stress.csv", index=False)
res["P1"] = {k: {c: [float(g[c].median()), float(g[c].quantile(.25)), float(g[c].quantile(.75))] for c in ["rest_fp", "stress_recall", "exercise_false_stress"]} | {"n_subjects": int(len(g))}
             for k, g in p1.groupby("features")}
for k, v in res["P1"].items(): print("P1", k, json.dumps(v))
# 運動誤判率是否高於靜息誤判率（配對）
for k, g in p1.groupby("features"):
    z = g.dropna(subset=["rest_fp", "exercise_false_stress"])
    res["P1"][k]["wilcoxon_ex_vs_rest_p"] = float(wilcoxon(z.exercise_false_stress, z.rest_fp).pvalue)
    print("P1 wilcoxon exercise vs rest false-positive", k, res["P1"][k]["wilcoxon_ex_vs_rest_p"])

# ---------- P2 ----------
rows = []
for s, g in lp.groupby("subject"):
    y = (g.label == "stress").astype(int).values
    if len(set(y)) < 2: continue
    rows.append(dict(subject=s, lr=balanced_accuracy_score(y, (g.p_LogReg >= .5).astype(int)), rf=balanced_accuracy_score(y, (g.p_RandomForest >= .5).astype(int))))
p2 = pd.DataFrame(rows); d = p2.lr - p2.rf
rng = np.random.default_rng(0); bs = [d.sample(len(d), replace=True, random_state=int(rng.integers(1e9))).mean() for _ in range(2000)]
res["P2"] = dict(n=len(p2), mean_diff_LR_minus_RF=float(d.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], wilcoxon_p=float(wilcoxon(p2.lr, p2.rf).pvalue))
print("P2", json.dumps(res["P2"]))

# ---------- P3 ----------
rows = []
for s, g in lp.groupby("subject"):
    est = g[g.task.isin(["Rest1", "TMCT"])]
    if not {"rest", "stress"} <= set(est.label): continue
    react = est[est.label == "stress"].hr_mean.mean() - est[est.label == "rest"].hr_mean.mean()
    ev = g[g.task.isin(["RealOpinion", "OppositeOpinion", "Subtract", "Rest2"])]
    y = (ev.label == "stress").astype(int).values
    if len(set(y)) < 2: continue
    rows.append(dict(subject=s, react_indep=react, balAcc_indep=balanced_accuracy_score(y, (ev.p_LogReg >= .5).astype(int)), n_stress_eval=int(y.sum())))
p3 = pd.DataFrame(rows); p3.to_csv(f"{OUT}/p3_cross_fitted_reactivity.csv", index=False)
r = spearmanr(p3.react_indep, p3.balAcc_indep)
res["P3"] = dict(n=len(p3), spearman=[float(r.statistic), float(r.pvalue)])
print("P3 independent-task HR reactivity vs accuracy: n=%d rho=%.3f p=%.4f" % (len(p3), r.statistic, r.pvalue))

# ---------- P4 ----------
try:
    import statsmodels.formula.api as smf
    m = pd.read_csv(Sp); m = m[~m.task.isin(["Rest1", "Rest2"])].dropna(subset=["self_stress", "hr", "rmssd"]).copy()
    m["hr_w"] = m.hr - m.groupby("subject").hr.transform("mean"); m["rm_w"] = m.rmssd - m.groupby("subject").rmssd.transform("mean")
    out = {}
    for var in ["hr_w", "rm_w"]:
        mod = smf.mixedlm(f"self_stress ~ {var} + C(task)", m, groups=m["subject"]).fit()
        out[var] = dict(coef=float(mod.params[var]), se=float(mod.bse[var]), p=float(mod.pvalues[var]), n=int(len(m)), n_subj=int(m.subject.nunique()))
    res["P4"] = out; print("P4 mixed model (stress-task phases only, task fixed effect, subject random intercept):", json.dumps(out))
except Exception as e:
    res["P4"] = f"failed: {e}"; print("P4 failed", e)

# ---------- P5 ----------
c = core.copy(); y = (c.label == "stress").astype(int).values
for name, mk in [("LogReg", lambda: LogisticRegression(max_iter=2000, class_weight="balanced")),
                 ("RandomForest", lambda: RandomForestClassifier(200, min_samples_leaf=3, class_weight="balanced", random_state=0, n_jobs=4))]:
    pred = np.zeros(len(c), int)
    for tr, te in StratifiedKFold(10, shuffle=True, random_state=0).split(c, y):
        Xa, Xb = prep(c.iloc[tr], c.iloc[te], FULL); pred[te] = mk().fit(Xa, y[tr]).predict(Xb)
    res[f"P5|random10fold|{name}"] = float(balanced_accuracy_score(y, pred))
    print(f"P5 random window-level 10-fold {name}: balAcc={res[f'P5|random10fold|{name}']:.3f} (subject-wise LOSO: LogReg 0.677 / RF 0.627)")
json.dump(res, open(f"{OUT}/priority_results.json", "w"), indent=1, default=float)
