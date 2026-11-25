"""延伸分析（Hongn 2025 公開資料，僅用腕錶也有的訊號 HR / HRV / 加速度）。
A. 留一任務驗證（leave-one-stressor-out，同時留出受試者）
B. 誰被誤判：每位受試者的辨識表現 vs 其生理反應幅度
C. 自評壓力（0-10）與生理量的受試者內關係
D. 選擇性預測（低信心時不判斷）：risk-coverage
用法：python 04_extended_analyses.py <windows.csv> <Stress_Level_v1.csv> <Stress_Level_v2.csv> <輸出資料夾>
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score

W, SL1, SL2, OUT = sys.argv[1:5]
df = pd.read_csv(W)
df["hrv_missing"] = df.rmssd.isna().astype(float)
COLS = ["hr_mean", "hr_std", "hr_slope", "rmssd", "sdnn", "meannn", "acc_mean", "acc_std", "acc_p95", "hrv_missing"]
st = df[df.label.isin(["rest", "stress"])].reset_index(drop=True)
y_all = (st.label == "stress").astype(int).values
subj_all = st.subject.values

def prep(tr, te):
    med = tr[COLS].median(); a, b = tr[COLS].fillna(med), te[COLS].fillna(med)
    mu, sd = a.mean(), a.std().replace(0, 1)
    return ((a - mu) / sd).values, ((b - mu) / sd).values

def fit_predict(tr, te, ytr, model):
    Xa, Xb = prep(tr, te)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced") if model == "LogReg" else RandomForestClassifier(200, min_samples_leaf=3, class_weight="balanced", random_state=0, n_jobs=4)
    clf.fit(Xa, ytr); return clf.predict_proba(Xb)[:, 1]

def boot_ci(subj, y, p, thr=0.5, n=1000, seed=0):
    rng = np.random.default_rng(seed); subs = np.unique(subj); idx = {s: np.where(subj == s)[0] for s in subs}; v = []
    for _ in range(n):
        ii = np.concatenate([idx[s] for s in rng.choice(subs, len(subs), replace=True)])
        if len(set(y[ii])) > 1: v.append(balanced_accuracy_score(y[ii], (p[ii] >= thr).astype(int)))
    return float(balanced_accuracy_score(y, (p >= thr).astype(int))), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

res = {}
# ---------- 基準 LOSO（機率）----------
for model in ["LogReg", "RandomForest"]:
    p = np.zeros(len(st))
    for s in np.unique(subj_all):
        te = subj_all == s
        p[te] = fit_predict(st[~te], st[te], y_all[~te], model)
    st[f"p_{model}"] = p
    res[f"LOSO|{model}"] = boot_ci(subj_all, y_all, p)
    print("LOSO", model, "balAcc=%.3f CI[%.3f,%.3f]" % res[f"LOSO|{model}"]); sys.stdout.flush()

# ---------- A. 留一任務（同時留出受試者）----------
tasks = ["Stroop", "TMCT", "RealOpinion", "OppositeOpinion", "Subtract"]
for model in ["LogReg", "RandomForest"]:
    for T in tasks:
        is_T = (st.task == T).values
        test_pool = ((st.label == "rest").values | is_T)           # 測試：同一人的靜息 + 被留出任務的壓力視窗
        keep_idx, p_loto = [], []
        for s in np.unique(subj_all[is_T]):
            te = test_pool & (subj_all == s)
            tr = (subj_all != s) & ~is_T                            # 訓練：其他人、且不含任務 T 的壓力視窗（含其他壓力任務與靜息）
            if te.sum() == 0 or len(set(y_all[te])) < 2 or len(set(y_all[tr])) < 2: continue
            p_loto.append(fit_predict(st[tr], st[te], y_all[tr], model)); keep_idx.append(np.where(te)[0])
        if not keep_idx: continue
        ii = np.concatenate(keep_idx); pl = np.concatenate(p_loto)
        base = st.loc[ii, f"p_{model}"].values                      # 同一批測試視窗的一般 LOSO 結果（訓練含任務 T）
        r_loto = boot_ci(subj_all[ii], y_all[ii], pl); r_base = boot_ci(subj_all[ii], y_all[ii], base)
        res[f"LOTO|{model}|{T}"] = dict(heldout_task=r_loto, ordinary_LOSO=r_base, n_stress=int(y_all[ii].sum()), n_subjects=int(len(np.unique(subj_all[ii]))))
        print(f"LOTO {model:12s} {T:16s} held-out-task balAcc={r_loto[0]:.3f} [{r_loto[1]:.3f},{r_loto[2]:.3f}] | ordinary LOSO same windows={r_base[0]:.3f} [{r_base[1]:.3f},{r_base[2]:.3f}] | n_stress={int(y_all[ii].sum())} n_subj={len(np.unique(subj_all[ii]))}"); sys.stdout.flush()

# ---------- B. 誰被誤判 ----------
rows = []
for s, g in st.groupby("subject"):
    yy = (g.label == "stress").astype(int).values
    if len(set(yy)) < 2: continue
    pr = (g["p_LogReg"].values >= 0.5).astype(int)
    d_hr = g[g.label == "stress"].hr_mean.mean() - g[g.label == "rest"].hr_mean.mean()
    d_rm = np.nanmean(np.log(g[g.label == "stress"].rmssd)) - np.nanmean(np.log(g[g.label == "rest"].rmssd))
    rows.append(dict(subject=s, stage=g.stage.iloc[0], balAcc=balanced_accuracy_score(yy, pr), d_hr=d_hr, d_logrmssd=d_rm,
                     n_stress=int(yy.sum()), n_rest=int((1 - yy).sum())))
subj = pd.DataFrame(rows); subj.to_csv(f"{OUT}/per_subject.csv", index=False)
r1 = spearmanr(subj.balAcc, subj.d_hr, nan_policy="omit"); r2 = spearmanr(subj.balAcc, subj.d_logrmssd, nan_policy="omit")
res["B_per_subject"] = dict(n=len(subj), balAcc_median=float(subj.balAcc.median()), balAcc_IQR=[float(subj.balAcc.quantile(.25)), float(subj.balAcc.quantile(.75))],
                            share_below_0_5=float((subj.balAcc < 0.5).mean()), share_above_0_7=float((subj.balAcc > 0.7).mean()),
                            spearman_balAcc_vs_dHR=[float(r1.statistic), float(r1.pvalue)], spearman_balAcc_vs_dlogRMSSD=[float(r2.statistic), float(r2.pvalue)])
print("B per-subject:", json.dumps(res["B_per_subject"])); sys.stdout.flush()

# ---------- C. 自評壓力 ----------
def load_sl(path, ver):
    x = pd.read_csv(path, index_col=0)
    ren = {"First Rest": "Rest1", "Second Rest": "Rest2", "Real Opinion": "RealOpinion", "Opposite Opinion": "OppositeOpinion"}
    x = x.rename(columns=ren).drop(columns=["Baseline"], errors="ignore")
    x = x.reset_index().rename(columns={x.index.name or "index": "subject"}).melt(id_vars="subject", var_name="task", value_name="self_stress")
    x["stage"] = ver; return x
sl = pd.concat([load_sl(SL1, "v1"), load_sl(SL2, "v2")], ignore_index=True)
ph = st.groupby(["subject", "stage", "task"]).agg(hr=("hr_mean", "mean"), rmssd=("rmssd", lambda v: np.nanmean(np.log(v)) if v.notna().any() else np.nan),
                                               acc=("acc_std", "mean"), n_win=("hr_mean", "size")).reset_index()
m = ph.merge(sl, on=["subject", "stage", "task"], how="inner").dropna(subset=["self_stress", "hr"])
m["hr_c"] = m.hr - m.groupby("subject").hr.transform("mean"); m["self_c"] = m.self_stress - m.groupby("subject").self_stress.transform("mean")
m["rm_c"] = m.rmssd - m.groupby("subject").rmssd.transform("mean"); m["acc_c"] = m.acc - m.groupby("subject").acc.transform("mean")
m.to_csv(f"{OUT}/phase_selfreport.csv", index=False)
def sp(a, b):
    k = m[[a, b]].dropna(); r = spearmanr(k[a], k[b]); return [float(r.statistic), float(r.pvalue), int(len(k))]
# 受試者層級 bootstrap：以受試者內中心化後的 Spearman
def boot_sp(a, b, n=1000, seed=0):
    rng = np.random.default_rng(seed); subs = m.subject.unique(); g = {s: m[m.subject == s][[a, b]].dropna() for s in subs}; v = []
    for _ in range(n):
        k = pd.concat([g[s] for s in rng.choice(subs, len(subs), replace=True)])
        if len(k) > 10 and k[a].nunique() > 1 and k[b].nunique() > 1: v.append(spearmanr(k[a], k[b]).statistic)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
res["C_selfreport_within_person"] = {k: sp(k, "self_c") + boot_sp(k, "self_c") for k in ["hr_c", "rm_c", "acc_c"]}
res["C_n_phases"] = int(len(m)); res["C_n_subjects"] = int(m.subject.nunique())
print("C self-report within-person Spearman [rho,p,n,CIlo,CIhi]:", json.dumps(res["C_selfreport_within_person"])); sys.stdout.flush()

# ---------- D. 選擇性預測 ----------
sel = {}
for model in ["LogReg", "RandomForest"]:
    p = st[f"p_{model}"].values; conf = np.abs(p - 0.5); out = []
    for cov in [1.0, 0.8, 0.6, 0.4, 0.2]:
        thr = np.quantile(conf, 1 - cov); keep = conf >= thr
        if len(set(y_all[keep])) < 2: continue
        out.append([cov, float(balanced_accuracy_score(y_all[keep], (p[keep] >= 0.5).astype(int))), int(keep.sum())])
    sel[model] = out; print("D risk-coverage", model, [(c, round(b, 3), n) for c, b, n in out])
res["D_selective"] = sel
json.dump(res, open(f"{OUT}/extended_results.json", "w"), indent=1, default=float)
st.to_csv(f"{OUT}/loso_probs.csv", index=False)
