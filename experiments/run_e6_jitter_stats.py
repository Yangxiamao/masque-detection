# -*- coding: utf-8 -*-
"""[2026-09-08] E6-B: timing 通道在观测点抖动下还剩多少 —— 不用 NetMamba 的独立验证。

## 为什么需要这个实验

E3 的 timing-only 条件给了 93.85%, 而 iat_statistical_baseline.py 的归属分析显示
RF 的头部特征重要度【全部落在 <100 us 的间隔占比】上 (frac_<10us / frac_10-100us
合计约占一半)。本项目的抓包点是【客户端本机】的 tshark, 2-3 us 量级的包间隔基本
只能来自 NIC 的 GRO / 中断合并 / 本机 socket 批处理, 而非网络可观测的结构。

但论文 Introduction 写的受益方是"运营商的策略执行与流量工程", 那是网络侧观测点。
于是有一个论文没回答的问题:

    timing 的充分性是不是抓包点的伪影? 换到网络中途的观测点还成立吗?

本脚本对到达时刻施加零均值高斯抖动 (见 simulate_obfuscation_bi.make_tf_jitter:
抖动 -> 重排 -> 重新差分), 逐个 sigma 重跑与 iat_statistical_baseline.py 完全
同构的四个模型族, 并观察特征重要度是否从 <100 us 迁走。

四个模型族一起报, 是为了让结论不依赖单一架构的归纳偏置 —— 与论文
"four model families, all far above the majority class" 的论证方式一致。

    wsl -e bash -lic 'conda activate MAMBA; cd /mnt/d/zemi/2026/dataset;
                      python experiments/run_e6_jitter_stats.py'
"""
import json
import os

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_ROOT = os.environ.get("MASQUE_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
PROC = os.path.join(_ROOT, "data", "processed")
OUT = os.path.join(_ROOT, "outputs", "e6_jitter")

# [2026-10-12 ICOIN r2 #3] MASQUE_RELABEL=1 时改标 11 条并输出到 *_relabel (见 pipeline/relabel_util.py)
import sys as _sys  # noqa: E402
_sys.path.insert(0, os.path.join(_ROOT, "pipeline"))
import relabel_util  # noqa: E402
OUT = relabel_util.out_dir(OUT)
os.makedirs(OUT, exist_ok=True)

SEED = 42
SEQ_LEN = 20          # 与 NetMamba+ 实际看到的长度一致
FLOOR = 1e-6
DECADE_EDGES = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]

# (标签, 数据集目录, sigma 秒)。sigma=0 即原始客户端观测点。
CONDITIONS = [
    ("client (sigma=0)", "quic_vs_masque_bi_feat", 0.0),
    ("sigma=50us",       "e6_jit50us",             50e-6),
    ("sigma=200us",      "e6_jit200us",            200e-6),
    ("sigma=1ms",        "e6_jit1ms",              1e-3),
    ("sigma=5ms",        "e6_jit5ms",              5e-3),
]

FEAT_NAMES = (
    ["log_mean", "log_std", "log_min", "log_max", "log_iqr"]
    + [f"log_q{q}" for q in (10, 25, 50, 75, 90)]
    + ["frac_<10us", "frac_10-100us", "frac_100us-1ms", "frac_1-10ms",
       "frac_10-100ms", "frac_0.1-1s", "frac_>1s"]
    + ["duration", "pkt_rate"]
)
SUB100US_IDX = [FEAT_NAMES.index("frac_<10us"), FEAT_NAMES.index("frac_10-100us")]

MODELS = {
    "RandomForest": lambda: RandomForestClassifier(n_estimators=500, random_state=SEED),
    "GradientBoosting": lambda: GradientBoostingClassifier(random_state=SEED),
    "LogisticRegression": lambda: make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=5000, random_state=SEED)),
}


def load_all(name):
    """合并 train/valid/test = 全 504 条; 按 pcap 文件名排序保证条件间逐条对齐。"""
    items = []
    for sp in ["train", "valid", "test"]:
        with open(os.path.join(PROC, name, f"data-{sp}.json"), encoding="utf-8") as f:
            items += json.load(f)
    items.sort(key=lambda it: os.path.basename(it["pcap_file"]))
    return relabel_util.relabel_items(items)


def featurize(item, limit=SEQ_LEN):
    vals = [float(v) for v in item["intervals"].split()][:limit]
    x = np.asarray(vals[1:], dtype=np.float64)
    if x.size == 0:
        return np.zeros(len(FEAT_NAMES))
    lg = np.log10(np.maximum(x, FLOOR))
    qs = np.percentile(lg, [10, 25, 50, 75, 90])
    hist = np.histogram(x, bins=[0.0] + DECADE_EDGES + [np.inf])[0] / x.size
    return np.concatenate([
        [lg.mean(), lg.std(), lg.min(), lg.max(), qs[3] - qs[1]], qs, hist,
        [x.sum(), x.size / max(x.sum(), 1e-9)],
    ])


def main():
    ref_keys = None
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    print(f"{'='*96}")
    print("E6-B  timing-only 在观测点抖动下的可分性 (全 504 会话, 5 折 CV, IAT 统计量)")
    print(f"{'='*96}")
    print(f"{'观测点':<20}{'RandomForest':>18}{'GradBoost':>14}{'LogReg':>14}"
          f"{'RF AUC':>10}{'<100us 重要度':>16}")
    print("-" * 96)

    for tag, dirname, sigma in CONDITIONS:
        items = load_all(dirname)
        keys = [os.path.basename(it["pcap_file"]) for it in items]
        if ref_keys is None:
            ref_keys = keys
        elif keys != ref_keys:
            raise SystemExit(f"{dirname}: 样本集合/顺序与基准条件不一致, 无法逐条对比")

        X = np.vstack([featurize(it) for it in items])
        y = np.asarray([it["label"] for it in items])

        row = {"sigma_s": sigma, "n": int(y.size), "models": {}}
        cells = []
        for name, factory in MODELS.items():
            sc = cross_val_score(factory(), X, y, cv=cv, scoring="accuracy")
            row["models"][name] = {"cv_mean": round(float(sc.mean()) * 100, 2),
                                   "cv_std": round(float(sc.std()) * 100, 2)}
            cells.append(f"{sc.mean()*100:.2f}±{sc.std()*100:.2f}")

        # RF 的 OOF 概率 -> AUC (accuracy 之外给一个阈值无关的量)
        prob = cross_val_predict(MODELS["RandomForest"](), X, y, cv=cv,
                                 method="predict_proba")[:, 1]
        auc = roc_auc_score(y, prob)
        row["rf_oof_auc"] = round(float(auc), 4)

        # <100us 两个 bin 的重要度之和: 判别力是否真的从微秒尺度迁走
        rf = MODELS["RandomForest"]().fit(X, y)
        imp = rf.feature_importances_
        sub = float(imp[SUB100US_IDX].sum())
        row["importance_sub100us"] = round(sub, 4)
        row["top_features"] = {FEAT_NAMES[i]: round(float(imp[i]), 4)
                               for i in np.argsort(imp)[::-1][:5]}

        print(f"{tag:<20}{cells[0]:>18}{cells[1]:>14}{cells[2]:>14}"
              f"{auc:>10.4f}{sub*100:>15.1f}%")
        results[tag] = row

    print("-" * 96)
    print(f"{'多数类基线':<20}{'53.57':>18}")

    print(f"\n各观测点下 RF 的头部特征 (看判别力迁到哪里去了):")
    for tag, row in results.items():
        feats = ", ".join(f"{k} {v:.3f}" for k, v in row["top_features"].items())
        print(f"  {tag:<20} {feats}")

    base = results["client (sigma=0)"]["models"]["RandomForest"]["cv_mean"]
    print(f"\n判读:")
    for tag, row in results.items():
        if tag == "client (sigma=0)":
            continue
        d = row["models"]["RandomForest"]["cv_mean"] - base
        print(f"  {tag:<12} RF {row['models']['RandomForest']['cv_mean']:.2f}% "
              f"({d:+.2f} pp vs 客户端观测点)")

    results["_meta"] = {
        "note": "抖动 = 到达时刻加零均值高斯噪声后重排再差分; 见 "
                "pipeline/pcap_split/simulate_obfuscation_bi.py:make_tf_jitter",
        "seq_len": SEQ_LEN,
        "majority_baseline": round(270 / 504 * 100, 2),
        "cv": "StratifiedKFold(5, shuffle=True, random_state=42) over all 504 sessions",
        "caveat": "每会话 1 条流, 故 5 折 CV 即会话级; 但折间站点重叠, 不度量跨站点泛化",
    }
    path = os.path.join(OUT, "summary_e6_jitter_stats.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nsaved: {path}")


if __name__ == "__main__":
    main()
