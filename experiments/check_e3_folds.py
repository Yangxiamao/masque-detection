"""[2026-08-23] E3 自检 S1-S4 (docs/01_Project_Management/E3_handoff.md §8)。

造折错了后面 15-95 次训练全废, 且**不会报错** —— 折跨条件错开时 McNemar 会静默失效。
所以在跑任何训练之前必须过这一关。

    python experiments/check_e3_folds.py --conditions full,nosizes_nobytes,c123
"""

import argparse
import json
import os
from collections import Counter

_ROOT = os.environ.get("MASQUE_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOLD_ROOT = os.path.join(_ROOT, "data", "processed", "e3_folds")

_ok, _fail = [], []


def check(name, cond, detail=""):
    (_ok if cond else _fail).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")


def load(cond, k, split):
    p = os.path.join(FOLD_ROOT, cond, f"fold{k}", f"data-{split}.json")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def keys(items):
    return [os.path.basename(it["pcap_file"]) for it in items]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", default="full,nosizes_nobytes,c123")
    ap.add_argument("--n_splits", type=int, default=5)
    args = ap.parse_args()
    conds = [c.strip() for c in args.conditions.split(",")]
    K = args.n_splits

    print(f"条件: {conds}\n折数: {K}\n")

    ref = conds[0]
    test_keys = {k: set(keys(load(ref, k, "test"))) for k in range(K)}
    all_items = load(ref, 0, "train") + load(ref, 0, "valid") + load(ref, 0, "test")
    label_of = {os.path.basename(it["pcap_file"]): it["label"] for it in all_items}
    N = len(label_of)
    full_ratio = Counter(label_of.values())

    # ---- S1: 5 折 test 互不相交, 并集 = 全集 ----
    print(f"S1  5 折 test 互不相交, 并集 = 全部 {N} 条")
    union = set()
    overlap = False
    for k in range(K):
        if union & test_keys[k]:
            overlap = True
        union |= test_keys[k]
    check("S1a 折间 test 无重叠", not overlap)
    check("S1b test 并集 == 全集", union == set(label_of),
          f"|union|={len(union)}, N={N}")

    # ---- S2: 分层比例 + 三集合无交集 ----
    print(f"\nS2  每折 test 类别比例与全集一致 (±2 条内); train/valid/test 无交集")
    print(f"    全集: Direct {full_ratio[0]} / Masque {full_ratio[1]} "
          f"(Masque {full_ratio[1]/N*100:.2f}%)")
    for k in range(K):
        te = test_keys[k]
        c = Counter(label_of[x] for x in te)
        exp0 = full_ratio[0] * len(te) / N
        exp1 = full_ratio[1] * len(te) / N
        check(f"S2a fold{k} 分层比例", abs(c[0] - exp0) <= 2 and abs(c[1] - exp1) <= 2,
              f"Direct {c[0]} (期望 {exp0:.1f}) / Masque {c[1]} (期望 {exp1:.1f})")
    for k in range(K):
        tr, va, te = (set(keys(load(ref, k, s))) for s in ["train", "valid", "test"])
        check(f"S2b fold{k} 三集合两两无交集",
              not (tr & va) and not (tr & te) and not (va & te),
              f"|tr&va|={len(tr&va)} |tr&te|={len(tr&te)} |va&te|={len(va&te)}")

    # ---- S3: 折跨条件逐条相同 (McNemar 的前提) ----
    print(f"\nS3  条件之间同一 fold 的 test/valid/train 列表逐条相同")
    for cond in conds[1:]:
        for k in range(K):
            for split in ["test", "valid", "train"]:
                a = keys(load(ref, k, split))
                b = keys(load(cond, k, split))
                same = a == b  # 逐条相同, 含顺序 (dump 预测靠顺序对齐)
                check(f"S3 {cond} vs {ref} fold{k} {split}", same,
                      "" if same else f"len {len(a)} vs {len(b)}, 首个差异位置见输出")

    # ---- S4: 每折三集合之和 = N, valid 非空 ----
    print(f"\nS4  每折 train+valid+test = {N}, valid 非空")
    for cond in conds:
        for k in range(K):
            n = {s: len(load(cond, k, s)) for s in ["train", "valid", "test"]}
            tot = sum(n.values())
            check(f"S4 {cond} fold{k}", tot == N and n["valid"] > 0,
                  f"train {n['train']} + valid {n['valid']} + test {n['test']} = {tot}")

    print(f"\n{'='*70}")
    print(f"通过 {len(_ok)} 项, 失败 {len(_fail)} 项")
    if _fail:
        print("失败项:")
        for f in _fail:
            print(f"  - {f}")
        raise SystemExit(1)
    print("自检全部通过, 可以开跑训练。")


if __name__ == "__main__":
    main()
