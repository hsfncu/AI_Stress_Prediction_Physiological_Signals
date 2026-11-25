"""ChatGPT 第二輪建議：PRV 缺失旗標與 PRV 特徵的敏感度分析（29 位有壓力、靜息、運動資料者；留一受試者）。
三種特徵組：
  full        = HR + PRV + 加速度 + no-PRV 旗標
  no_flag     = HR + PRV + 加速度（去掉旗標）
  hr_acc      = HR + 加速度（去掉所有 PRV 特徵與旗標）
每組評估：rest-only 訓練（壓力召回、靜息誤判、運動誤判）與 三類別訓練（同上）。
用法：python 10_prv_flag_sensitivity.py <windows.csv> <輸出資料夾>
"""
import sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

Wp, OUT = sys.argv[1:3]
df = pd.read_csv(Wp); df["hrv_missing"] = df.rmssd.isna().astype(float)
SETS = {
    "full": ["hr_mean", "hr_std", "hr_slope", "rmssd", "sdnn", "meannn", "acc_mean", "acc_std", "acc_p95", "hrv_missing"],
    "no_flag": ["hr_mean", "hr_std", "hr_slope", "rmssd", "sdnn", "meannn", "acc_mean", "acc_std", "acc_p95"],
    "hr_acc": ["hr_mean", "hr_std", "hr_slope", "acc_mean", "acc_std", "acc_p95"],
}
subs = sorted(set(df[df.label == "exercise"].subject) & set(df[df.label == "stress"].subject) & set(df[df.label == "rest"].subject))
lab3 = {"rest": 0, "stress": 1, "exercise": 2}

def prep(tr, te, cols):
    med = tr[cols].median(); a, b = tr[cols].fillna(med), te[cols].fillna(med); mu, sd = a.mean(), a.std().replace(0, 1)
    return ((a - mu) / sd).values, ((b - mu) / sd).values

def mk(model):
    return LogisticRegression(max_iter=3000, class_weight="balanced") if model == "LogReg" else RandomForestClassifier(200, min_samples_leaf=3, class_weight="balanced", random_state=0, n_jobs=4)

rows = []
for model in ["LogReg", "RandomForest"]:
    for fs, cols in SETS.items():
        for s in subs:
            tr = df[df.subject != s]; te = df[df.subject == s]; lab = te.label.values
            # rest-only
            trb = tr[tr.label.isin(["rest", "stress"])]
            Xa, Xb = prep(trb, te, cols); pr = mk(model).fit(Xa, (trb.label == "stress").astype(int).values).predict(Xb)
            rows.append(dict(model=model, features=fs, training="rest-only", subject=s, stress_recall=float((pr[lab == "stress"] == 1).mean()), rest_as_stress=float((pr[lab == "rest"] == 1).mean()), exercise_as_stress=float((pr[lab == "exercise"] == 1).mean())))
            # three-class
            Xa, Xb = prep(tr, te, cols); pr = mk(model).fit(Xa, tr.label.map(lab3).values).predict(Xb)
            rows.append(dict(model=model, features=fs, training="three-class", subject=s, stress_recall=float((pr[lab == "stress"] == 1).mean()), rest_as_stress=float((pr[lab == "rest"] == 1).mean()), exercise_as_stress=float((pr[lab == "exercise"] == 1).mean())))
r = pd.DataFrame(rows); r.to_csv(f"{OUT}/p1e_prv_flag_sensitivity.csv", index=False)
res = {}
for (m, fs, tr), g in r.groupby(["model", "features", "training"]):
    q = lambda c: f"{g[c].median():.2f} ({g[c].quantile(.25):.2f}-{g[c].quantile(.75):.2f})"
    res[f"{m}|{fs}|{tr}"] = {c: [float(g[c].median()), float(g[c].quantile(.25)), float(g[c].quantile(.75))] for c in ["stress_recall", "rest_as_stress", "exercise_as_stress"]}
    print(f"{m:12s} {fs:8s} {tr:12s} n={len(g)} | stress recall {q('stress_recall')} | rest->stress {q('rest_as_stress')} | exercise->stress {q('exercise_as_stress')}")
json.dump(res, open(f"{OUT}/p1e_results.json", "w"), indent=1)
