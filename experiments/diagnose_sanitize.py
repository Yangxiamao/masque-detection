"""[2026-10-12 ICOIN r2 #1] 字节通道泄漏诊断: raw vs sanitized 的逐包 / 逐流可预测性。

问题: tf_sanitize_bytes 抹掉了 IP/UDP 头与长度字段, 但 QUIC 头 (DCID/SCID 长度、
CID、包类型) 保留, 它们按方向 (也按类别) 取不同值。这里量化剩下多少:
  逐包: 由 320 B 预测 方向 (up/dn)、包长档 (L<240 / 240<=L<1200 / L>=1200)
  逐流: 由 5x320 B 预测 类别 (Direct / MASQUE)
分类器 = RandomForest(300), 5 折复用 outputs/e3_kfold/fold_assignment.json 的 test 划分
(逐包任务按流分组, 同一条流的包不跨折)。WSL MAMBA 环境运行:
    python experiments/diagnose_sanitize.py
输出 outputs/icoin_r2/sanitize_leak_diag.json
"""

import glob
import json
import os
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score

_ROOT = os.environ.get("MASQUE_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "pipeline", "pcap_split"))
import sanitize_bytes as S  # noqa: E402

FOLD_ROOT = os.path.join(_ROOT, "data", "processed", "e3_folds", "full")
OUT = os.path.join(_ROOT, "outputs", "icoin_r2", "sanitize_leak_diag.json")


def load():
    flows = []
    for k in range(5):
        with open(os.path.join(FOLD_ROOT, f"fold{k}", "data-test.json"), encoding="utf-8") as f:
            for it in json.load(f):
                flows.append((k, it))
    return flows


def pkt_rows(flows, sanitize):
    X, ydir, ylen, yflow_cls, fold = [], [], [], [], []
    for k, it in flows:
        src = S.tf_sanitize_bytes(it) if sanitize else it
        sizes = [int(s) for s in it["signed_sizes"].split()]
        for i, d in enumerate(src["data"]):
            orig = [int(x) for x in it["data"][i].split()]
            L = ((orig[2] << 8) | orig[3]) - 28
            X.append([int(x) for x in d.split()])
            ydir.append(1 if sizes[i] > 0 else 0)
            ylen.append(0 if L < 240 else (1 if L < 1200 else 2))
            yflow_cls.append(it["label"])
            fold.append(k)
    return (np.array(X, dtype=np.uint8), np.array(ydir), np.array(ylen),
            np.array(yflow_cls), np.array(fold))


def flow_rows(flows, sanitize):
    X, y, fold = [], [], []
    for k, it in flows:
        src = S.tf_sanitize_bytes(it) if sanitize else it
        X.append([int(x) for d in src["data"] for x in d.split()])
        y.append(it["label"])
        fold.append(k)
    return np.array(X, dtype=np.uint8), np.array(y), np.array(fold)


def cv(X, y, fold, mask=None):
    pred = np.empty_like(y)
    for k in range(5):
        tr, te = fold != k, fold == k
        clf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0)
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te])
    sel = np.ones_like(y, dtype=bool) if mask is None else mask
    return {"acc": float((pred[sel] == y[sel]).mean()),
            "balanced_acc": float(balanced_accuracy_score(y[sel], pred[sel])),
            "n": int(sel.sum()),
            "class_counts": {int(c): int((y[sel] == c).sum()) for c in np.unique(y[sel])}}


def main():
    flows = load()
    res = {}
    for name, san in [("raw", False), ("sanitized", True)]:
        X, ydir, ylen, ycls, fold = pkt_rows(flows, san)
        r = {
            "pkt_direction": cv(X, ydir, fold),
            "pkt_direction_within_Direct": cv(X, ydir, fold, ycls == 0),
            "pkt_direction_within_MASQUE": cv(X, ydir, fold, ycls == 1),
            "pkt_length_bucket": cv(X, ylen, fold),
            "pkt_header_region_only_direction": cv(X[:, :80], ydir, fold),
            "pkt_payload_region_only_direction": cv(X[:, 80:], ydir, fold),
        }
        Xf, yf, ff = flow_rows(flows, san)
        r["flow_class"] = cv(Xf, yf, ff)
        res[name] = r
        print(name, json.dumps(r, indent=1))
    res["note"] = ("RF(300) 逐包 5 折 (按流分组, 复用 e3 fold_assignment 的 test 划分)。"
                   "within_* 的 mask 只作用于评估, 训练仍用全部包。"
                   "sanitized = pipeline/pcap_split/sanitize_bytes.tf_sanitize_bytes")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print("->", OUT)


if __name__ == "__main__":
    main()
