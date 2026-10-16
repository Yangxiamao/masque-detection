#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
E3 追加解析：把 McNemar 配对检验从 4 组补全到「全部条件 vs Full」。

背景
----
`run_kfold_e3.py` 只算了 4 组配对比较（full vs c123 / nosizes_nobytes /
no_direction，以及 c123 vs c123_nobytes）。论文的消融表要给每一行一个 p 值，
否则读者无法判断「No-Intervals 98.02% vs Full 98.02%」是真的没差还是没测。

本脚本不训练、不碰 GPU，只读 `outputs/e3_kfold/<cond>/fold{0..4}/preds.json`
里已落盘的逐样本预测重算。前提是所有条件共用同一份 `fold_assignment.json`
（E3 已用 `check_e3_folds.py` 的 S1–S4 共 57 项断言核验过），因此条件之间是
配对的。本脚本自己再断言一次样本集合与标签逐条一致，不信任上游。

检验方式与 E3 主脚本一致：`binomtest(b, b+c, 0.5)` 精确检验（b+c 小，卡方
近似不可靠），多重比较用 Holm 校正。校正的族 = 本次全部比较。

用法
----
    python experiments/analyze_e3_mcnemar.py
    python experiments/analyze_e3_mcnemar.py --baseline full --out <path.json>

输出
----
`outputs/e3_kfold/mcnemar_full_matrix.json` + 一张打印出来的表。
"""

import argparse
import json
import os
import sys

from scipy.stats import binomtest

_ROOT = os.environ.get("MASQUE_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
E3_DIR = os.path.join(_ROOT, "outputs", "e3_kfold")

# 论文消融表与常量消融表用到的条件（seed 0 的主运行；_s1/_s2 是 Tier C 的重复种子，
# 不进配对检验的族，否则同一条件会被重复计入多重校正）
CONDITIONS = [
    "full",
    "no_intervals",
    "no_direction",
    "no_bytes",
    "no_sizes",
    "nosizes_nobytes",
    "c1",
    "c2",
    "c3",
    "c123",
    "c123_nobytes",
]


def load_oof(condition, n_folds=5, cond_dir=None):
    """把 5 折的 preds.json 拼成 {basename: (pred, label)}，即 out-of-fold 全集。"""
    oof = {}
    for k in range(n_folds):
        path = os.path.join(cond_dir or os.path.join(E3_DIR, condition), f"fold{k}", "preds.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"缺少逐样本预测：{path}")
        with open(path, "r", encoding="utf-8") as f:
            for name, rec in json.load(f).items():
                if name in oof:
                    raise ValueError(f"{condition}: {name} 在多个折的 test 里出现")
                oof[name] = (int(rec["pred"]), int(rec["label"]))
    return oof


def mcnemar(correct_a, correct_b, keys):
    """返回 (b, c, p)。b = 仅 A 对，c = 仅 B 对。"""
    b = sum(1 for k in keys if correct_a[k] and not correct_b[k])
    c = sum(1 for k in keys if correct_b[k] and not correct_a[k])
    if b + c == 0:
        return b, c, 1.0  # 逐条预测完全一致
    return b, c, binomtest(b, b + c, 0.5).pvalue


def holm(pvals):
    """Holm-Bonferroni 校正，返回与输入等长的校正后 p（保序、上限 1.0）。"""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)  # 保证单调不减
        adjusted[i] = min(1.0, running)
    return adjusted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="full", help="配对检验的参照条件")
    # [2026-10-12 ICOIN r2 §7.5] 改标 + 新选模规则的族: --e3_dir 指向 outputs/e3r_kfold_sel,
    # --extra 追加不在 e3 目录下的条件 (表 I 的 No-Length 在 icoin_r2/... 下)。默认值不变。
    ap.add_argument("--e3_dir", default=E3_DIR, help="条件目录的根 (默认 outputs/e3_kfold)")
    ap.add_argument("--extra", action="append", default=[],
                    help="追加进族的条件, 形如 name=<目录>(含 fold0..4/preds.json), 可重复")
    ap.add_argument("--out", default=None,
                    help="默认 <e3_dir>/mcnemar_full_matrix.json")
    args = ap.parse_args()
    e3_dir = args.e3_dir if os.path.isabs(args.e3_dir) else os.path.join(_ROOT, args.e3_dir)
    out = args.out or os.path.join(e3_dir, "mcnemar_full_matrix.json")
    if os.path.abspath(e3_dir) != os.path.abspath(E3_DIR) and \
            os.path.abspath(out) == os.path.abspath(os.path.join(E3_DIR, "mcnemar_full_matrix.json")):
        sys.exit("拒绝用其他目录的结果覆盖 outputs/e3_kfold/mcnemar_full_matrix.json")
    extra = {}
    for spec in args.extra:
        name, d = spec.split("=", 1)
        extra[name] = d if os.path.isabs(d) else os.path.join(_ROOT, d)
    CONDITIONS.extend(extra)

    oofs = {}
    for cond in CONDITIONS:
        try:
            oofs[cond] = load_oof(cond, cond_dir=extra.get(cond, os.path.join(e3_dir, cond)))
        except FileNotFoundError as e:
            print(f"[skip] {e}")

    if args.baseline not in oofs:
        sys.exit(f"参照条件 {args.baseline} 的预测不存在")

    # --- 自检：所有条件的样本集合与标签必须逐条相同，否则配对无效 -------------
    base_keys = set(oofs[args.baseline])
    for cond, oof in oofs.items():
        if set(oof) != base_keys:
            sys.exit(f"[FAIL] {cond} 的样本集合与 {args.baseline} 不同 -> 配对检验无效")
        for k in base_keys:
            if oof[k][1] != oofs[args.baseline][k][1]:
                sys.exit(f"[FAIL] {cond} 的 {k} 标签与 {args.baseline} 不一致")
    keys = sorted(base_keys)
    print(f"[ok] {len(oofs)} 个条件、{len(keys)} 条流，样本集合与标签逐条一致（配对成立）")

    correct = {
        cond: {k: (oof[k][0] == oof[k][1]) for k in keys} for cond, oof in oofs.items()
    }

    comparisons = []
    for cond in CONDITIONS:
        if cond not in oofs or cond == args.baseline:
            continue
        b, c, p = mcnemar(correct[args.baseline], correct[cond], keys)
        comparisons.append(
            {
                "A": args.baseline,
                "B": cond,
                "acc_A": sum(correct[args.baseline].values()) / len(keys),
                "acc_B": sum(correct[cond].values()) / len(keys),
                "b_only_A_correct": b,
                "c_only_B_correct": c,
                "n_discordant": b + c,
                "p": p,
            }
        )

    for comp, p_adj in zip(comparisons, holm([c["p"] for c in comparisons])):
        comp["p_holm"] = p_adj

    print(
        f"\n{'条件':<20}{'OOF acc':>9}{'b':>5}{'c':>5}{'p':>11}{'p (Holm)':>11}  判读"
    )
    print("-" * 74)
    print(f"{args.baseline:<20}{sum(correct[args.baseline].values())/len(keys)*100:>8.2f}%"
          f"{'—':>5}{'—':>5}{'—':>11}{'—':>11}  (参照)")
    for comp in comparisons:
        verdict = "显著" if comp["p_holm"] < 0.05 else "无显著差异"
        if comp["n_discordant"] <= 2:
            verdict += "（检出力极低）"
        print(
            f"{comp['B']:<20}{comp['acc_B']*100:>8.2f}%"
            f"{comp['b_only_A_correct']:>5}{comp['c_only_B_correct']:>5}"
            f"{comp['p']:>11.4g}{comp['p_holm']:>11.4g}  {verdict}"
        )

    payload = {
        "baseline": args.baseline,
        "n": len(keys),
        "family_size": len(comparisons),
        "correction": "holm",
        "test": "exact McNemar via scipy.stats.binomtest(b, b+c, 0.5)",
        "comparisons": comparisons,
        "caveat": (
            "不显著 != 等价。discordant 数很小时检出力极低，"
            "不能反过来当作『该通道无用』的证据。"
        ),
    }
    if os.path.abspath(e3_dir) != os.path.abspath(E3_DIR) or extra:
        payload["e3_dir"] = os.path.relpath(e3_dir, _ROOT)
        payload["extra_conditions"] = {k: os.path.relpath(v, _ROOT) for k, v in extra.items()}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n已写出：{out}")


if __name__ == "__main__":
    main()
