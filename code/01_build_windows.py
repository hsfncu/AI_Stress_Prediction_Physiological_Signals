"""Hongn 2025 (PhysioNet wearable-device-dataset 1.0.1) 壓力/運動/靜息 辨別：特徵建立 + LOSO 基準。
只用手錶也有的訊號：HR、IBI(HRV)、加速度。EDA/TEMP 不用於主分析。
標籤依 notebook 的 tag 區段定義（見 STRESS_SPANS 註解）。
"""
import os, sys, glob, warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

ROOT = glob.glob(sys.argv[1] + "/wearable-device*/Wearable_Dataset")[0]
OUT = sys.argv[2]
W = 30  # 秒
STEP = 5  # LSTM 內部時間步長（秒）

def t0_of(d, fn="EDA.csv"):
    return pd.Timestamp(open(f"{d}/{fn}").readline().strip().split(",")[0])

def load_session(d):
    t0 = t0_of(d)
    hr_t0 = t0_of(d, "HR.csv")
    hr = np.loadtxt(f"{d}/HR.csv", delimiter=",", skiprows=2)
    hr_t = (hr_t0 - t0).total_seconds() + np.arange(len(hr))
    ibi_raw = pd.read_csv(f"{d}/IBI.csv", skiprows=1, header=None).values if os.path.getsize(f"{d}/IBI.csv") > 40 else np.zeros((0, 2))
    acc = np.loadtxt(f"{d}/ACC.csv", delimiter=",", skiprows=2) / 64.0  # g
    acc_mag = np.sqrt((acc ** 2).sum(1))
    acc_t = np.arange(len(acc)) / 32.0
    tags = [pd.Timestamp(x) for x in open(f"{d}/tags.csv").read().split("\n") if x.strip()]
    tags_s = [0.0] + [(x - t0).total_seconds() for x in tags]
    return dict(hr=hr, hr_t=hr_t, ibi=ibi_raw, acc=acc_mag, acc_t=acc_t, tags=tags_s)

def spans_for(sid, cond, tg):
    """回傳 [(start,end,label)]。label: stress / rest / exercise。
    STRESS v1(S*)：壓力=[3,4],[5,6],[7,8],[9,10],[11,12]；靜息=[4,5],[6,7]（Stroop、TMCT、真實意見、反向意見、減法；兩段休息）
    STRESS v2(f*)：壓力=[2,3],[4,5],[6,7],[8,9]；靜息=[3,4],[7,8]
    （對應 notebook 的 axvspan 索引；各段長度與協定相符：意見/減法約 30 秒，休息數分鐘）
    運動：第一個到最後一個 tag 之間。"""
    out = []
    if cond == "STRESS":
        if sid.startswith("S"):
            st = [(3, 4, "Stroop"), (5, 6, "TMCT"), (7, 8, "RealOpinion"), (9, 10, "OppositeOpinion"), (11, 12, "Subtract")]
            rs = [(4, 5, "Rest1"), (6, 7, "Rest2")]
        else:
            st = [(2, 3, "TMCT"), (4, 5, "RealOpinion"), (6, 7, "OppositeOpinion"), (8, 9, "Subtract")]
            rs = [(3, 4, "Rest1"), (7, 8, "Rest2")]
        need = max(j for _, j, _ in st + rs)
        if len(tg) <= need: return []
        out += [(tg[i], tg[j], "stress", nm) for i, j, nm in st] + [(tg[i], tg[j], "rest", nm) for i, j, nm in rs]
    else:
        if len(tg) < 3: return []
        out.append((tg[1], tg[-1], "exercise", cond.title()))
    return out

def feats(s, a, b):
    m = (s["hr_t"] >= a) & (s["hr_t"] < b)
    hr = s["hr"][m]
    if len(hr) < W * 0.7: return None
    ib = s["ibi"]; mi = (ib[:, 0] + 0 >= a) & (ib[:, 0] < b) if len(ib) else np.array([], bool)
    nn = ib[mi, 1] if len(ib) else np.array([])
    nn = nn[(nn > 0.3) & (nn < 2.0)]
    if len(nn) >= 8:
        d = np.diff(nn); rmssd = np.sqrt((d ** 2).mean()); sdnn = nn.std(); meannn = nn.mean()
    else:
        rmssd = sdnn = meannn = np.nan
    ma = (s["acc_t"] >= a) & (s["acc_t"] < b)
    ac = s["acc"][ma]
    if len(ac) < 32 * W * 0.7: return None
    f = dict(hr_mean=hr.mean(), hr_std=hr.std(), hr_slope=np.polyfit(np.arange(len(hr)), hr, 1)[0] if len(hr) > 3 else 0,
             rmssd=rmssd, sdnn=sdnn, meannn=meannn, acc_mean=ac.mean(), acc_std=ac.std(), acc_p95=np.percentile(ac, 95))
    # LSTM 序列：每 STEP 秒一步 [hr, acc_std, acc_mean]
    seq = []
    for k in range(W // STEP):
        a2, b2 = a + k * STEP, a + (k + 1) * STEP
        h = s["hr"][(s["hr_t"] >= a2) & (s["hr_t"] < b2)]
        c = s["acc"][(s["acc_t"] >= a2) & (s["acc_t"] < b2)]
        seq.append([h.mean() if len(h) else np.nan, c.std() if len(c) else np.nan, c.mean() if len(c) else np.nan])
    return f, np.array(seq)

rows, seqs = [], []
inventory = []
for cond in ["STRESS", "AEROBIC", "ANAEROBIC"]:
    for d in sorted(glob.glob(f"{ROOT}/{cond}/*")):
        sid = os.path.basename(d)
        reason = None
        if "_" in sid: reason = "split_file"
        if cond == "STRESS" and sid in ("S02", "f07"): reason = "known_issue(S02 duplicated signal / f07 PPG covered)"
        if reason: inventory.append((cond, sid, reason)); continue
        s = load_session(d)
        sp = spans_for(sid, cond, s["tags"])
        if not sp: inventory.append((cond, sid, "no_tags")); continue
        n = 0
        for a, b, lab, task in sp:
            a2, b2 = a + (W if cond != "STRESS" else 0), b - (W if cond != "STRESS" else 0)
            t = a2
            while t + W <= b2:
                r = feats(s, t, t + W)
                if r is not None:
                    f, q = r
                    rows.append(dict(subject=sid, stage="v1" if sid.startswith("S") else "v2", cond=cond, label=lab, task=task, t=t, **f)); seqs.append(q); n += 1
                t += W
        inventory.append((cond, sid, f"ok:{n}win"))
df = pd.DataFrame(rows); X_seq = np.array(seqs)
df.to_csv(f"{OUT}/windows.csv", index=False); np.save(f"{OUT}/seq.npy", X_seq)
pd.DataFrame(inventory, columns=["cond", "subject", "status"]).to_csv(f"{OUT}/inventory.csv", index=False)
print(df.groupby(["label"]).size().to_string())
print("subjects with each label:", df.groupby("label").subject.nunique().to_dict())
print("subjects with all 3 labels:", sum(1 for _, g in df.groupby("subject") if g.label.nunique() == 3))
print("nan frac feats:", df[["rmssd", "sdnn"]].isna().mean().round(3).to_dict())
