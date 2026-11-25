import os, sys, warnings, json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
import torch, torch.nn as nn
torch.set_num_threads(4)
OUT = sys.argv[1]
df = pd.read_csv(f"{OUT}/windows.csv"); SEQ = np.load(f"{OUT}/seq.npy")
df["hrv_missing"] = df.rmssd.isna().astype(float)
SETS = {
    "full(HR+HRV+ACC+missing)": ["hr_mean","hr_std","hr_slope","rmssd","sdnn","meannn","acc_mean","acc_std","acc_p95","hrv_missing"],
    "no_missing_flag": ["hr_mean","hr_std","hr_slope","rmssd","sdnn","meannn","acc_mean","acc_std","acc_p95"],
    "no_ACC(HR+HRV+missing)": ["hr_mean","hr_std","hr_slope","rmssd","sdnn","meannn","hrv_missing"],
    "HR_only": ["hr_mean","hr_std","hr_slope"],
    "ACC_only": ["acc_mean","acc_std","acc_p95"],
    "missing_flag_only": ["hrv_missing"],
}
TASKS = {"stress_vs_rest": ["rest","stress"], "stress_vs_exercise": ["exercise","stress"]}

def prep(Xtr, Xte):
    med = Xtr.median(); Xtr, Xte = Xtr.fillna(med), Xte.fillna(med)
    mu, sd = Xtr.mean(), Xtr.std().replace(0, 1)
    return ((Xtr-mu)/sd).values, ((Xte-mu)/sd).values

class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.l = nn.LSTM(3,16,batch_first=True); s.o = nn.Linear(16,2)
    def forward(s,x): h,_ = s.l(x); return s.o(h[:,-1])

def lstm_pred(itr, ite, ytr, seed):
    a, b = SEQ[itr].copy(), SEQ[ite].copy()
    m = np.nanmedian(a.reshape(-1,3),0)
    for x in (a,b):
        idx = np.where(np.isnan(x)); x[idx] = np.take(m, idx[2])
    mu, sd = a.reshape(-1,3).mean(0), a.reshape(-1,3).std(0)+1e-6
    a, b = (a-mu)/sd, (b-mu)/sd
    torch.manual_seed(seed); net = Net(); opt = torch.optim.Adam(net.parameters(), 3e-3)
    cnt = np.bincount(ytr, minlength=2).astype(float); w = torch.tensor(len(ytr)/(2*np.maximum(cnt,1)), dtype=torch.float32)
    lf = nn.CrossEntropyLoss(weight=w); X = torch.tensor(a, dtype=torch.float32); y = torch.tensor(ytr)
    for _ in range(40):
        p = torch.randperm(len(X))
        for i in range(0,len(X),128):
            bb = p[i:i+128]; opt.zero_grad(); lf(net(X[bb]), y[bb]).backward(); opt.step()
    with torch.no_grad(): return net(torch.tensor(b, dtype=torch.float32)).argmax(1).numpy()

def ci(sub_ids, y, p, n=1000, seed=0):
    rng = np.random.default_rng(seed); subs = np.unique(sub_ids)
    idx = {s: np.where(sub_ids==s)[0] for s in subs}
    vals = []
    for _ in range(n):
        pick = rng.choice(subs, len(subs), replace=True)
        ii = np.concatenate([idx[s] for s in pick])
        if len(set(y[ii]))<2: continue
        vals.append(balanced_accuracy_score(y[ii], p[ii]))
    return float(balanced_accuracy_score(y,p)), float(np.percentile(vals,2.5)), float(np.percentile(vals,97.5))

res = {}
for task, classes in TASKS.items():
    sub = df[df.label.isin(classes)]; y = sub.label.map({c:i for i,c in enumerate(classes)}).values
    ids = sub.index.values; subj = sub.subject.values
    for fs, cols in SETS.items():
        for mname in ["LogReg","RandomForest"]:
            pred = np.full(len(sub), -1)
            for s in np.unique(subj):
                te = subj==s; tr = ~te
                Xtr, Xte = prep(sub.loc[tr, cols], sub.loc[te, cols])
                clf = LogisticRegression(max_iter=2000, class_weight="balanced") if mname=="LogReg" else RandomForestClassifier(200, min_samples_leaf=3, class_weight="balanced", random_state=0, n_jobs=4)
                pred[te] = clf.fit(Xtr, y[tr]).predict(Xte)
            r = ci(subj, y, pred); res[f"{task}|{fs}|{mname}"] = r
            print(f"{task:19s} {fs:26s} {mname:12s} balAcc={r[0]:.3f} 95%CI[{r[1]:.3f},{r[2]:.3f}]"); sys.stdout.flush()
    # LSTM, 3 seeds, full sequence only
    for seed in [0,1,2]:
        pred = np.full(len(sub), -1)
        for s in np.unique(subj):
            te = subj==s; tr = ~te
            pred[te] = lstm_pred(ids[tr], ids[te], y[tr], seed)
        r = ci(subj, y, pred); res[f"{task}|LSTM_seed{seed}"] = r
        print(f"{task:19s} LSTM seed{seed:<22d} balAcc={r[0]:.3f} 95%CI[{r[1]:.3f},{r[2]:.3f}]"); sys.stdout.flush()
json.dump(res, open(f"{OUT}/robust.json","w"), indent=1)
