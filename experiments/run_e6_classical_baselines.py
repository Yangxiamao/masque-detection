# -*- coding: utf-8 -*-
"""[2026-09-08] E6-H: 论文引用的两个"经典分类器"数字的可复现复算。

论文里有两个不来自 NetMamba+ 的数字被当作关键论据:

  * "logistic regression on the raw 100-dimensional (signed size, interval)
     vector reaches 98.8% under 5-fold cross-validation" —— 用来论证"任务不需要
     深模型"。追溯下来它只出现在 2026-07 的汇报材料里 (docs/04_.../current_wiki.md
     等), 仓库里【没有生成它的脚本】, 也没有对应的 outputs/*.json, 而主稿头部
     声称每个数字都能映射到一个产出文件。
  * "a random forest on 24 packet-length summary statistics reaches
     99.21% ± 0.74%" —— 用作 length 通道充分性的证据。

本脚本把两者用同一套协议重算并落盘: 全 504 条会话、分层 5 折 CV (每会话恰好
一条流, 故折即会话级)、固定种子。顺带给出 ROC-AUC 与 0.5 阈值上的 TPR/FPR,
因为论文全篇只报 accuracy。

    wsl -e bash -lic 'conda activate MAMBA; cd /mnt/d/zemi/2026/dataset;
                      python experiments/run_e6_classical_baselines.py'
"""
import json
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_ROOT = os.environ.get("MASQUE_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(_ROOT, "data", "processed", "quic_vs_masque_bi_feat")
OUT = os.path.join(_ROOT, "outputs", "e6_classical")

# [2026-10-12 ICOIN r2 #3] MASQUE_RELABEL=1 时改标 11 条并输出到 *_relabel (见 pipeline/relabel_util.py)
import sys as _sys  # noqa: E402
_sys.path.insert(0, os.path.join(_ROOT, "pipeline"))
import relabel_util  # noqa: E402
OUT = relabel_util.out_dir(OUT)
SEED = 42


def load_all():
    items = []
    for sp in ["train", "valid", "test"]:
        with open(os.path.join(DATA, f"data-{sp}.json"), encoding="utf-8") as f:
            items += json.load(f)
    items.sort(key=lambda it: os.path.basename(it["pcap_file"]))
    return relabel_util.relabel_items(items)


def raw_vector(it, limit=None):
    """原始拼接向量: signed size 与 interval 直接首尾相接, 不做任何统计加工。"""
    s = [float(v) for v in it["signed_sizes"].split()][:limit]
    v = [float(v) for v in it["intervals"].split()][:limit]
    return np.asarray(s + v, dtype=np.float64)


def length_stats(it, limit=None):
    """24 个包长统计量: 全部包 / 上行 / 下行 各 8 个 (max min mean std q25 q50 q75 count)。"""
    s = np.asarray([float(v) for v in it["signed_sizes"].split()][:limit])
    out = []
    for sub in (np.abs(s), s[s > 0], -s[s < 0]):
        if sub.size == 0:
            out += [0.0] * 8
            continue
        out += [sub.max(), sub.min(), sub.mean(), sub.std(),
                *np.percentile(sub, [25, 50, 75]), float(sub.size)]
    return np.asarray(out, dtype=np.float64)


# ⚠️ 输入宽度必须显式对齐。NetMamba+ 的 seq_len=20, 只看前 20 个包; 而既有的
# run_const_rf_baseline.py 里那个 99.21% 的长度 RF 用的是【全部 50 个包】
# (sizes_of() 不截断)。两者不同输入, 直接并列会高估经典基线。故每个特征都跑
# 20 包与 50 包两档, 论文里做跨模型族比较时只能引 20 包那一档。
RF = lambda: RandomForestClassifier(n_estimators=500, random_state=SEED)   # noqa: E731
LR = lambda: make_pipeline(StandardScaler(),                                # noqa: E731
                           LogisticRegression(max_iter=5000, random_state=SEED))

MODELS = {}
for _w in (20, 50):
    MODELS[f"raw_first{_w}_LogReg"] = (lambda it, w=_w: raw_vector(it, w), LR)
    MODELS[f"raw_first{_w}_RandomForest"] = (lambda it, w=_w: raw_vector(it, w), RF)
    MODELS[f"len24_first{_w}_RandomForest"] = (lambda it, w=_w: length_stats(it, w), RF)
    MODELS[f"len24_first{_w}_LogReg"] = (lambda it, w=_w: length_stats(it, w), LR)


def main():
    os.makedirs(OUT, exist_ok=True)
    items = load_all()
    y = np.asarray([it["label"] for it in items])
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    print(f"n = {len(items)} 条会话 (MASQUE {n_pos} / Direct {n_neg}), "
          f"多数类基线 {max(n_pos, n_neg)/len(y)*100:.2f}%\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    res = {}
    print(f"{'模型':<24}{'5 折 CV acc':>18}{'OOF acc':>10}{'AUC':>9}"
          f"{'TPR':>8}{'FPR':>8}")
    print("-" * 78)
    for name, (feat, factory) in MODELS.items():
        X = np.vstack([feat(it) for it in items])
        sc = cross_val_score(factory(), X, y, cv=cv, scoring="accuracy")
        pred = cross_val_predict(factory(), X, y, cv=cv)
        prob = cross_val_predict(factory(), X, y, cv=cv, method="predict_proba")[:, 1]
        acc = float((pred == y).mean())
        auc = float(roc_auc_score(y, prob))
        tpr = float(((pred == 1) & (y == 1)).sum() / n_pos)
        fpr = float(((pred == 1) & (y == 0)).sum() / n_neg)
        print(f"{name:<24}{sc.mean()*100:>12.2f}±{sc.std()*100:.2f}{acc*100:>9.2f}%"
              f"{auc:>9.4f}{tpr*100:>7.1f}%{fpr*100:>7.1f}%")
        res[name] = {"n_features": int(X.shape[1]),
                     "cv_mean": round(float(sc.mean()) * 100, 2),
                     "cv_std": round(float(sc.std()) * 100, 2),
                     "oof_acc": round(acc * 100, 2), "auc": round(auc, 4),
                     "tpr": round(tpr, 4), "fpr": round(fpr, 4)}

    res["_meta"] = {
        "n": len(items), "n_masque": n_pos, "n_direct": n_neg,
        "cv": "StratifiedKFold(5, shuffle=True, random_state=42) over all 504 sessions",
        "note": "每会话恰好 1 条流, 故 5 折即会话级切分; 折间站点重叠, 不度量跨站点泛化",
        "purpose": "为论文引用的 raw-100 LogReg 与 24-stat RF 两个数字提供可复现产出",
    }
    path = os.path.join(OUT, "summary_classical_baselines.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\nsaved: {path}")


if __name__ == "__main__":
    main()
