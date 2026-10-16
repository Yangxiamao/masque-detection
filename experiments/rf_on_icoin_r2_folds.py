"""[2026-10-12 ICOIN r2 #1] 换一个自适应攻击者: 在 run_icoin_r2.py 的同一批折数据上训练 RandomForest。

输入与 NetMamba+ 看到的完全相同: 前 5 包 x 320 B 字节 + 前 20 包 signed size + 前 20 包 interval。
每折在 train+valid 上训练, 在 test 上评估, 拼成 504 条 OOF。另报字节位置的重要度 top-10,
以确认 sanitized 版的残留信号来自哪里。WSL MAMBA 环境:
    python experiments/rf_on_icoin_r2_folds.py --labels orig [--only t3dir_raw,t3dir_san]
输出 outputs/icoin_r2/<labels>/rf_summary.json
"""

import argparse
import json
import math
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier

_ROOT = os.environ.get("MASQUE_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONDS = ["t3dir_raw", "t3dir_san", "t3dir_strict", "nolen_pad", "nolen_pad_nobytes", "nolen_pad_san",
         "t3dir_c123_san", "t3dir_c123_hdr", "t3dir_c123_hdrtype"]


def feats(it):
    b = [int(x) for d in it["data"][:5] for x in d.split()]
    s = [int(x) for x in it["signed_sizes"].split()[:20]]
    iv = [float(x) for x in it["intervals"].split()[:20]]
    return b + s + iv


def wilson(k, n, z=1.959963984540054):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [(c - h) / d, (c + h) / d]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    root = os.path.join(_ROOT, "data", "processed", "icoin_r2_folds", args.labels)
    conds = args.only.split(",") if args.only else CONDS
    out = {}
    for c in conds:
        if not os.path.isdir(os.path.join(root, c)):
            continue
        cm = [[0, 0], [0, 0]]
        imp = np.zeros(5 * 320 + 40)
        per_fold = []
        for k in range(5):
            d = os.path.join(root, c, f"fold{k}")
            tr = []
            for s in ("train", "valid"):
                with open(os.path.join(d, f"data-{s}.json"), encoding="utf-8") as f:
                    tr += json.load(f)
            with open(os.path.join(d, "data-test.json"), encoding="utf-8") as f:
                te = json.load(f)
            Xtr = np.array([feats(i) for i in tr]); ytr = np.array([i["label"] for i in tr])
            Xte = np.array([feats(i) for i in te]); yte = np.array([i["label"] for i in te])
            clf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0).fit(Xtr, ytr)
            pr = clf.predict(Xte)
            for y, p in zip(yte, pr):
                cm[y][p] += 1
            per_fold.append(float((pr == yte).mean()))
            imp += clf.feature_importances_
        n = sum(map(sum, cm)); ok = cm[0][0] + cm[1][1]
        top = np.argsort(-imp)[:10]
        def name(j):
            if j < 1600:
                pkt, off = divmod(j, 320)
                return f"pkt{pkt}_{'hdr' if off < 80 else 'pl'}{off if off < 80 else off - 80}"
            return f"size{j-1600}" if j < 1620 else f"iat{j-1620}"
        out[c] = {"acc": ok / n, "ci95": wilson(ok, n), "cm": cm, "per_fold": per_fold,
                  "top_importance": [[name(int(j)), float(imp[j] / 5)] for j in top]}
        print(c, f"{ok/n*100:.2f}%", cm, [round(a * 100, 1) for a in per_fold],
              out[c]["top_importance"][:5], flush=True)
    p = os.path.join(_ROOT, "outputs", "icoin_r2", args.labels, "rf_summary.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    old = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
    old.update(out)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(old, f, indent=2)
    print("->", p)


if __name__ == "__main__":
    main()
