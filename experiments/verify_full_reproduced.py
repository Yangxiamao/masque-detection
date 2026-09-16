# -*- coding: utf-8 -*-
"""[2026-09-08] 校验重跑 full 条件是否逐条复现了已发表的预测。

为什么需要: E6-D 的"干净训练"检测器与 E6-E 的 ROC 都需要 full 条件【全部 5 折】的
checkpoint 与后验概率, 而 2026-08-23 那次跑完就把 fold1-4 的 checkpoint 删了
(每个 22.5 MB)。只能重跑, 但重跑会覆盖 outputs/e3_kfold/full/*/preds.json ——
论文 Table I/III 的 McNemar 全部建立在这批预测上。

于是流程是: 先把原目录整体备份到 _backup_full_20260908, 重跑, 然后用本脚本逐条
比对 pred 与 label。相同则说明训练确定性成立、已发表数字未被动过 (新文件只是多了
一个 prob1 字段); 不同则必须回滚备份, 并且论文里要显式说明 ROC 用的是另一次训练。

    python experiments/verify_full_reproduced.py
"""
import json
import os
import sys

_ROOT = os.environ.get("MASQUE_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEW = os.path.join(_ROOT, "outputs", "e3_kfold", "full")
OLD = os.path.join(_ROOT, "outputs", "e3_kfold", "_backup_full_20260908")


def load(root):
    oof = {}
    for k in range(5):
        p = os.path.join(root, f"fold{k}", "preds.json")
        if not os.path.exists(p):
            sys.exit(f"缺文件: {p}")
        with open(p, encoding="utf-8") as f:
            oof.update(json.load(f))
    return oof


def main():
    a, b = load(OLD), load(NEW)
    if set(a) != set(b):
        sys.exit(f"样本集合不同: 旧 {len(a)}, 新 {len(b)}")

    diff_pred = [k for k in a if a[k]["pred"] != b[k]["pred"]]
    diff_lab = [k for k in a if a[k]["label"] != b[k]["label"]]
    has_prob = sum(1 for v in b.values() if "prob1" in v)

    acc_a = sum(v["pred"] == v["label"] for v in a.values()) / len(a) * 100
    acc_b = sum(v["pred"] == v["label"] for v in b.values()) / len(b) * 100
    print(f"n = {len(a)}   旧 acc {acc_a:.4f}%   新 acc {acc_b:.4f}%")
    print(f"prob1 字段: {has_prob}/{len(b)} 条")
    print(f"label 不一致: {len(diff_lab)} 条")
    print(f"pred  不一致: {len(diff_pred)} 条")

    if diff_lab:
        sys.exit(f"!!! label 不一致, 折分配被改动了: {diff_lab[:3]}")
    if diff_pred:
        print(f"\n⚠️ 预测不完全复现 ({len(diff_pred)} 条不同)。例: {diff_pred[:5]}")
        print("   -> 论文中已发表的 Table I/III 数字必须继续引用备份里的那一版;")
        print("      回滚: rm -rf outputs/e3_kfold/full && "
              "mv outputs/e3_kfold/_backup_full_20260908 outputs/e3_kfold/full")
        sys.exit(2)
    if has_prob != len(b):
        sys.exit("!!! 新预测缺 prob1 字段, ROC 无法计算")
    print("\n逐条复现通过: 已发表数字未变, 且新增了后验概率。")


if __name__ == "__main__":
    main()
