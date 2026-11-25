"""審稿意見後的補充檢查。
A. LSTM（3 種子）逐受試者預測，與邏輯迴歸做受試者層級配對比較
B. 三類別模型（靜息/壓力/運動，留一受試者，29 人）：每類召回；以及不加類別權重的運動納入二元訓練
C. 運動與壓力視窗落在「訓練受試者靜息+壓力視窗 5–95 百分位範圍外」的比率（每人，留一受試者建立參考範圍）
D. Holm 校正（自評壓力的兩個混合模型檢定）
用法：python 09_additional_checks.py <windows.csv> <loso_probs.csv> <輸出資料夾>
"""
import os, sys, json, warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
import torch, torch.nn as nn
torch.set_num_threads(4)

Wp, Lp, OUT = sys.argv[1:4]
df = pd.read_csv(Wp); df["hrv_missing"] = df.rmssd.isna().astype(float)
SEQ = np.load(os.path.join(os.path.dirname(Wp), "seq.npy"))
lp = pd.read_csv(Lp)
FULL = ["hr_mean", "hr_std", "hr_slope", "rmssd", "sdnn", "meannn", "acc_mean", "acc_std", "acc_p95", "hrv_missing"]
res = {}

def prep(tr, te, cols):
    med = tr[cols].median(); a, b = tr[cols].fillna(med), te[cols].fillna(med); mu, sd = a.mean(), a.std().replace(0, 1)
    return ((a - mu) / sd).values, ((b - mu) / sd).values

# ---------- A. LSTM paired vs LR ----------
core_idx = df.index[df.label.isin(["rest", "stress"])].values
core = df.loc[core_idx].reset_index(drop=True); y = (core.label == "stress").astype(int).values; subj = core.subject.values
class Net(nn.Module):
    def __init__(s): super().__init__(); s.l = nn.LSTM(3, 16, batch_first=True); s.o = nn.Linear(16, 2)
    def forward(s, x): h, _ = s.l(x); return s.o(h[:, -1])
def lstm_pred(itr, ite, ytr, seed):
    a, b = SEQ[itr].copy(), SEQ[ite].copy(); m = np.nanmedian(a.reshape(-1, 3), 0)
    for x in (a, b):
        idx = np.where(np.isnan(x)); x[idx] = np.take(m, idx[2])
    mu, sd = a.reshape(-1, 3).mean(0), a.reshape(-1, 3).std(0) + 1e-6; a, b = (a - mu) / sd, (b - mu) / sd
    torch.manual_seed(seed); net = Net(); opt = torch.optim.Adam(net.parameters(), 3e-3)
    cnt = np.bincount(ytr, minlength=2).astype(float); w = torch.tensor(len(ytr) / (2 * np.maximum(cnt, 1)), dtype=torch.float32); lf = nn.CrossEntropyLoss(weight=w)
    X = torch.tensor(a, dtype=torch.float32); Y = torch.tensor(ytr)
    for _ in range(40):
        p = torch.randperm(len(X))
        for i in range(0, len(X), 128):
            bb = p[i:i + 128]; opt.zero_grad(); lf(net(X[bb]), Y[bb]).backward(); opt.step()
    with torch.no_grad(): return net(torch.tensor(b, dtype=torch.float32)).argmax(1).numpy()
lr_pred = (lp.set_index(["subject", "t"]).loc[list(zip(core.subject, core.t))].p_LogReg.values >= 0.5).astype(int)
rng = np.random.default_rng(0)
def part_ba(pred):
    return pd.Series({s: balanced_accuracy_score(y[subj == s], pred[subj == s]) for s in np.unique(subj) if len(set(y[subj == s])) > 1})
lr_ba = part_ba(lr_pred)
A = {}
for seed in [0, 1, 2]:
    pred = np.zeros(len(core), int)
    for s in np.unique(subj):
        te = subj == s; pred[te] = lstm_pred(core_idx[~te], core_idx[te], y[~te], seed)
    ba = part_ba(pred); d = ba - lr_ba
    bs = [d.sample(len(d), replace=True, random_state=int(rng.integers(1e9))).mean() for _ in range(2000)]
    A[f"seed{seed}"] = dict(mean_diff_LSTM_minus_LR=float(d.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], wilcoxon_p=float(wilcoxon(ba, lr_ba).pvalue))
    print("A LSTM seed", seed, json.dumps(A[f"seed{seed}"])); sys.stdout.flush()
res["A_LSTM_vs_LR"] = A

# ---------- B. 三類別 & 無權重運動納入 ----------
subs = sorted(set(df[df.label == "exercise"].subject) & set(df[df.label == "stress"].subject) & set(df[df.label == "rest"].subject))
lab3 = {"rest": 0, "stress": 1, "exercise": 2}
rows = []
for model in ["LogReg", "RandomForest"]:
    for variant in ["3class_balanced", "binary_exerciseNeg_unweighted"]:
        for s in subs:
            tr = df[df.subject != s]; te = df[df.subject == s]
            Xa, Xb = prep(tr, te, FULL)
            if variant == "3class_balanced": ytr = tr.label.map(lab3).values
            else: ytr = (tr.label == "stress").astype(int).values
            cw = "balanced" if variant == "3class_balanced" else None
            clf = LogisticRegression(max_iter=3000, class_weight=cw) if model == "LogReg" else RandomForestClassifier(200, min_samples_leaf=3, class_weight=cw, random_state=0, n_jobs=4)
            pr = clf.fit(Xa, ytr).predict(Xb); lab = te.label.values
            if variant == "3class_balanced":
                rows.append(dict(model=model, variant=variant, subject=s, stress_recall=float((pr[lab == "stress"] == 1).mean()), rest_recall=float((pr[lab == "rest"] == 0).mean()), exercise_recall=float((pr[lab == "exercise"] == 2).mean()),
                                 rest_as_stress=float((pr[lab == "rest"] == 1).mean()), exercise_as_stress=float((pr[lab == "exercise"] == 1).mean())))
            else:
                rows.append(dict(model=model, variant=variant, subject=s, stress_recall=float((pr[lab == "stress"] == 1).mean()), rest_as_stress=float((pr[lab == "rest"] == 1).mean()), exercise_as_stress=float((pr[lab == "exercise"] == 1).mean())))
b = pd.DataFrame(rows); b.to_csv(f"{OUT}/p1c_threeclass_unweighted.csv", index=False)
B = {}
for (m, v), g in b.groupby(["model", "variant"]):
    B[f"{m}|{v}"] = {c: [round(float(g[c].median()), 3), round(float(g[c].quantile(.25)), 3), round(float(g[c].quantile(.75)), 3)] for c in g.columns if c not in ("model", "variant", "subject")}
    print("B", m, v, json.dumps(B[f"{m}|{v}"])); sys.stdout.flush()
res["B_threeclass_unweighted"] = B

# ---------- C. 訓練範圍外比率 ----------
feats = ["hr_mean", "acc_std", "hrv_missing"]; rows = []
for s in subs:
    ref = df[(df.subject != s) & df.label.isin(["rest", "stress"])]; te = df[df.subject == s]
    r = dict(subject=s)
    for f in ["hr_mean", "acc_std", "acc_mean"]:
        lo, hi = ref[f].quantile(.05), ref[f].quantile(.95)
        for st in ["rest", "stress", "exercise"]:
            v = te[te.label == st][f].dropna(); r[f"{f}|{st}"] = float(((v < lo) | (v > hi)).mean()) if len(v) else np.nan
    rows.append(r)
c = pd.DataFrame(rows); c.to_csv(f"{OUT}/p1d_out_of_range.csv", index=False)
C = {k: [round(float(c[k].median()), 3), round(float(c[k].quantile(.25)), 3), round(float(c[k].quantile(.75)), 3)] for k in c.columns if k != "subject"}
res["C_out_of_range"] = C; print("C out-of-range (outside 5-95th pct of other participants' rest+stress windows), median [Q1,Q3]:"); [print("  ", k, v) for k, v in C.items()]

# ---------- D. Holm ----------
ps = {"HR": 0.6472347, "lnRMSSD": 0.0467658}
order = sorted(ps, key=ps.get); holm = {}; prev = 0
for i, k in enumerate(order):
    adj = min(1.0, (len(order) - i) * ps[k]); adj = max(adj, prev); holm[k] = adj; prev = adj
res["D_holm"] = holm; print("D Holm-adjusted p:", holm)
json.dump(res, open(f"{OUT}/additional_checks.json", "w"), indent=1, default=float)
