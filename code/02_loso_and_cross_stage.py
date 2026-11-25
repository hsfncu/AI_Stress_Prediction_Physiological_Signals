import os, sys, warnings, json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix
import torch, torch.nn as nn
torch.set_num_threads(4)

OUT = sys.argv[1]
df = pd.read_csv(f"{OUT}/windows.csv"); SEQ = np.load(f"{OUT}/seq.npy")
FEATS = ["hr_mean", "hr_std", "hr_slope", "rmssd", "sdnn", "meannn", "acc_mean", "acc_std", "acc_p95"]
df["hrv_missing"] = df.rmssd.isna().astype(float)
COLS = FEATS + ["hrv_missing"]

def prep_tab(tr, te):
    Xtr, Xte = tr[COLS].copy(), te[COLS].copy()
    med = Xtr.median()
    Xtr, Xte = Xtr.fillna(med), Xte.fillna(med)
    mu, sd = Xtr.mean(), Xtr.std().replace(0, 1)
    return ((Xtr - mu) / sd).values, ((Xte - mu) / sd).values

def prep_seq(itr, ite):
    a, b = SEQ[itr].copy(), SEQ[ite].copy()
    m = np.nanmedian(a.reshape(-1, 3), 0)
    for x in (a, b):
        idx = np.where(np.isnan(x)); x[idx] = np.take(m, idx[2])
    mu, sd = a.reshape(-1, 3).mean(0), a.reshape(-1, 3).std(0) + 1e-6
    return (a - mu) / sd, (b - mu) / sd

class Net(nn.Module):
    def __init__(s, k):
        super().__init__(); s.l = nn.LSTM(3, 16, batch_first=True); s.o = nn.Linear(16, k)
    def forward(s, x):
        h, _ = s.l(x); return s.o(h[:, -1])

def fit_lstm(Xtr, ytr, Xte, k, seed=0, ep=40):
    torch.manual_seed(seed); np.random.seed(seed)
    net = Net(k); opt = torch.optim.Adam(net.parameters(), 3e-3)
    cnt = np.bincount(ytr, minlength=k).astype(float); w = torch.tensor(len(ytr) / (k * np.maximum(cnt, 1)), dtype=torch.float32)
    lossf = nn.CrossEntropyLoss(weight=w)
    X = torch.tensor(Xtr, dtype=torch.float32); y = torch.tensor(ytr)
    for _ in range(ep):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 128):
            b = perm[i:i + 128]; opt.zero_grad(); lossf(net(X[b]), y[b]).backward(); opt.step()
    with torch.no_grad():
        return net(torch.tensor(Xte, dtype=torch.float32)).argmax(1).numpy()

def models(Xtr, ytr, Xte, k, itr, ite):
    out = {}
    out["LogReg"] = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, ytr).predict(Xte)
    out["RandomForest"] = RandomForestClassifier(300, min_samples_leaf=3, class_weight="balanced", random_state=0, n_jobs=4).fit(Xtr, ytr).predict(Xte)
    st, se = prep_seq(itr, ite)
    out["LSTM"] = fit_lstm(st, ytr, se, k)
    out["Majority"] = np.full(len(Xte), np.bincount(ytr).argmax())
    return out

TASKS = {"3class": ["rest", "stress", "exercise"], "stress_vs_rest": ["rest", "stress"], "stress_vs_exercise": ["exercise", "stress"]}

def evaluate(task, train_idx_fn):
    classes = TASKS[task]; sub = df[df.label.isin(classes)]
    lab = sub.label.map({c: i for i, c in enumerate(classes)}).values
    ids = sub.index.values
    preds = {m: np.full(len(sub), -1) for m in ["LogReg", "RandomForest", "LSTM", "Majority"]}
    folds = train_idx_fn(sub)
    for tr_mask, te_mask in folds:
        if te_mask.sum() == 0: continue
        Xtr, Xte = prep_tab(sub[tr_mask], sub[te_mask])
        p = models(Xtr, lab[tr_mask], Xte, len(classes), ids[tr_mask], ids[te_mask])
        for m in p: preds[m][te_mask] = p[m]
    res = {}
    ok = preds["LogReg"] >= 0
    for m, p in preds.items():
        y, q = lab[ok], p[ok]
        # 受試者層級平衡準確度平均
        sb = []
        for s, g in sub[ok].groupby("subject"):
            yy, qq = lab[ok][(sub[ok].subject == s).values], p[ok][(sub[ok].subject == s).values]
            if len(set(yy)) > 1: sb.append(balanced_accuracy_score(yy, qq))
        res[m] = dict(macroF1=round(f1_score(y, q, average="macro"), 3), balAcc=round(balanced_accuracy_score(y, q), 3),
                      subjBalAcc_mean=round(float(np.mean(sb)), 3) if sb else None, n=int(ok.sum()),
                      confusion=confusion_matrix(y, q, labels=list(range(len(classes)))).tolist())
    return classes, res

def loso(sub):
    return [(sub.subject.values != s, sub.subject.values == s) for s in sub.subject.unique()]
def v1_to_v2(sub): return [((sub.stage == "v1").values, (sub.stage == "v2").values)]
def v2_to_v1(sub): return [((sub.stage == "v2").values, (sub.stage == "v1").values)]

report = {}
for name, fn in [("LOSO", loso), ("train_v1_test_v2", v1_to_v2), ("train_v2_test_v1", v2_to_v1)]:
    for task in TASKS:
        classes, res = evaluate(task, fn)
        report[f"{name}|{task}"] = dict(classes=classes, **res)
        print(f"== {name} | {task} {classes}")
        for m, r in res.items(): print(f"  {m:13s} macroF1={r['macroF1']}  balAcc={r['balAcc']}  subjBalAcc={r['subjBalAcc_mean']}  n={r['n']}  conf={r['confusion']}")
        sys.stdout.flush()
json.dump(report, open(f"{OUT}/results.json", "w"), indent=1)
