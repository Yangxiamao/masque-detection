# -*- coding: utf-8 -*-
"""[2026-09-08] E6-E: ROC、低 FPR 工作点, 以及现实先验下的精确率。

## 动机: 论文报的 accuracy 回答不了它自己在 Introduction 里提出的问题

Introduction 说这件事对运营商重要 —— 策略执行、流量工程、容量规划会把越来越多的
UDP/443 误归类。但真实链路上 MASQUE 的先验远不是本数据集的 53.6%; 更可能是百分之
几甚至千分之几。在那种 base rate 下, 98% 的 accuracy 完全不足以说明可用性:
FPR 3% 配上 1% 的先验, 精确率只有 25%。

论文全文没有一处 precision / recall / ROC / base rate 讨论。本脚本补上:
  (a) OOF 概率的 ROC-AUC;
  (b) 若干低 FPR 工作点上的 TPR;
  (c) 把 (TPR, FPR) 投影到若干假想先验下的精确率。

## 必须写进论文的分辨率上限

负类 (Direct) 只有 234 条, 所以【可分辨的最小非零 FPR 是 1/234 = 0.43%】。
低于此的工作点在本数据集上无法估计, 报"TPR@FPR=0.1%"是在编数字。这一条本身
就是一个诚实的局限声明, 比给一个无法支撑的数字有价值。

    python experiments/analyze_e6_operating_points.py --condition full
"""

import argparse
import json
import math
import os

_ROOT = os.environ.get("MASQUE_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E3 = os.path.join(_ROOT, "outputs", "e3_kfold")
OUT = os.path.join(_ROOT, "outputs", "e6_operating_points")

# [2026-10-12 ICOIN r2 #3] MASQUE_RELABEL=1 时改标 11 条并输出到 *_relabel (见 pipeline/relabel_util.py)
import sys as _sys  # noqa: E402
_sys.path.insert(0, os.path.join(_ROOT, "pipeline"))
import relabel_util  # noqa: E402
OUT = relabel_util.out_dir(OUT)
if relabel_util.RELABEL:
    E3 = os.path.join(_ROOT, "outputs", "e3r_kfold")   # 改标 OOF 的 preds.json 已带新标签
# [2026-10-12 ICOIN r2 §7.5] MASQUE_SELRULE=acc_loss -> 读同规则的 *_sel OOF, 输出加 _sel; 未设置时恒等
_sys.path.insert(0, os.path.join(_ROOT, "experiments"))
import selrule_util  # noqa: E402
E3 = selrule_util.out_dir(E3)
OUT = selrule_util.out_dir(OUT)

FPR_TARGETS = [0.10, 0.05, 0.02, 0.01]
PREVALENCES = [0.5357, 0.10, 0.01, 0.001]


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def load_oof(cond):
    oof = {}
    for k in range(5):
        p = os.path.join(E3, cond, f"fold{k}", "preds.json")
        if not os.path.exists(p):
            raise SystemExit(f"缺 {p}")
        with open(p, encoding="utf-8") as f:
            oof.update(json.load(f))
    if not all("prob1" in v for v in oof.values()):
        raise SystemExit(
            f"条件 {cond} 的 preds.json 里没有 prob1 字段。\n"
            f"这些是 2026-08-23 跑的, 那时 eval_dump_preds.py 还不落盘概率。\n"
            f"补跑: python experiments/run_kfold_e3.py --only {cond} --keep-ckpt --force")
    return oof


def roc(y, s):
    """返回按阈值降序的 (fpr, tpr, threshold) 列表 + AUC (梯形法)。"""
    pairs = sorted(zip(s, y), key=lambda t: -t[0])
    P = sum(y)
    N = len(y) - P
    pts, tp, fp = [(0.0, 0.0, math.inf)], 0, 0
    i = 0
    while i < len(pairs):
        thr = pairs[i][0]
        while i < len(pairs) and pairs[i][0] == thr:   # 同分样本必须一起进
            tp += pairs[i][1]
            fp += 1 - pairs[i][1]
            i += 1
        pts.append((fp / N, tp / P, thr))
    auc = 0.0
    for (f0, t0, _), (f1, t1, _) in zip(pts, pts[1:]):
        auc += (f1 - f0) * (t0 + t1) / 2
    return pts, auc, P, N


def tpr_at_fpr(pts, target):
    """FPR <= target 的最大 TPR (保守读法: 不在点之间插值)。"""
    best = (0.0, math.inf)
    for f, t, thr in pts:
        if f <= target + 1e-12 and t >= best[0]:
            best = (t, thr)
    return best


def precision_at(prev, tpr, fpr):
    num = prev * tpr
    den = num + (1 - prev) * fpr
    return num / den if den > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="full")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    oof = load_oof(args.condition)
    keys = sorted(oof)
    y = [oof[k]["label"] for k in keys]
    s = [oof[k]["prob1"] for k in keys]
    pts, auc, P, N = roc(y, s)

    # 0.5 阈值上的实际工作点 (= 论文报的 accuracy 所在的点)
    tp = sum(1 for k in keys if oof[k]["label"] == 1 and oof[k]["pred"] == 1)
    fp = sum(1 for k in keys if oof[k]["label"] == 0 and oof[k]["pred"] == 1)
    tpr0, fpr0 = tp / P, fp / N
    acc = sum(1 for k in keys if oof[k]["pred"] == oof[k]["label"]) / len(keys)

    print(f"{'='*84}")
    print(f"E6-E  条件 {args.condition}: OOF 工作点分析 (n={len(keys)}, "
          f"MASQUE {P} / Direct {N})")
    print(f"{'='*84}")
    print(f"ROC-AUC = {auc:.4f}")
    print(f"argmax 工作点 (阈值 0.5, 即论文引用的那个点):")
    lo, hi = wilson(tp, P)
    print(f"  accuracy {acc*100:.2f}%   TPR(recall) {tpr0*100:.2f}% [{lo*100:.2f}, {hi*100:.2f}]"
          f"   FPR {fpr0*100:.2f}% ({fp}/{N})")
    prec = precision_at(P / (P + N), tpr0, fpr0)
    print(f"  本数据集先验 ({P/(P+N)*100:.1f}%) 下的精确率 {prec*100:.2f}%")

    print(f"\n可分辨的最小非零 FPR = 1/{N} = {1/N*100:.2f}%  "
          f"—— 低于此的工作点本数据集无法估计")

    print(f"\n低 FPR 工作点:")
    print(f"{'目标 FPR':>10}{'实得 TPR':>12}{'阈值':>10}")
    print("-" * 34)
    rows = {}
    for tgt in FPR_TARGETS:
        t, thr = tpr_at_fpr(pts, tgt)
        rows[tgt] = {"tpr": t, "threshold": None if thr == math.inf else thr}
        thr_s = "inf" if thr == math.inf else f"{thr:.4f}"
        print(f"{tgt*100:>9.1f}%{t*100:>11.2f}%{thr_s:>10}")

    print(f"\n不同先验下的精确率 (取上表的工作点; 行 = 先验, 列 = 目标 FPR):")
    hdr = f"{'先验':>10}" + "".join(f"{f'FPR<={t*100:g}%':>13}" for t in FPR_TARGETS)
    print(hdr)
    print("-" * len(hdr))
    proj = {}
    for prev in PREVALENCES:
        line = f"{prev*100:>9.2f}%"
        proj[prev] = {}
        for tgt in FPR_TARGETS:
            pr = precision_at(prev, rows[tgt]["tpr"], tgt)
            proj[prev][tgt] = pr
            line += f"{pr*100:>12.1f}%"
        print(line)

    # 若要在 1% 先验下达到 90% 精确率, 需要多低的 FPR
    need = None
    for f, t, thr in pts:
        if t > 0 and precision_at(0.01, t, max(f, 1e-9)) >= 0.90:
            need = (f, t, thr)
            break
    print(f"\n1% 先验下要达到 90% 精确率所需的工作点:")
    if need and need[0] > 0:
        print(f"  FPR {need[0]*100:.3f}%  TPR {need[1]*100:.2f}%  —— "
              f"{'本数据集无法验证 (低于 1/234 分辨率)' if need[0] < 1/N else '数据集内可达'}")
    else:
        print(f"  需要 FPR < {1/N*100:.2f}%, 已低于本数据集的分辨率, 无法验证")

    res = {"condition": args.condition, "n": len(keys), "n_pos": P, "n_neg": N,
           "auc": round(auc, 4), "accuracy": round(acc, 4),
           "operating_point_0.5": {"tpr": tpr0, "fpr": fpr0,
                                   "precision_at_dataset_prior": prec},
           "min_resolvable_fpr": 1 / N,
           "tpr_at_fpr": {str(k): v for k, v in rows.items()},
           "precision_projection": {str(p): {str(t): v for t, v in d.items()}
                                    for p, d in proj.items()},
           "caveat": f"负类仅 {N} 条, 最小可分辨 FPR = {1/N:.4f}; "
                     f"更低的工作点不可估计。折间模型不独立, Wilson 区间偏窄。"}
    path = os.path.join(OUT, f"operating_points_{args.condition}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\nsaved: {path}")


if __name__ == "__main__":
    main()
