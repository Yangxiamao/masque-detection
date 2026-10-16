# -*- coding: utf-8 -*-
"""[2026-10-15 ICOIN r3 #4] 稳态段 (第 51-100 包) 的单通道充分性。

教授第 3 轮意见 #4: VI-A 的 99.33% 只是全特征模型; length / direction / timing 各自在稳态段
是否仍然充分, 此前没有验证。这里把 Table I 下半块的三个单通道条件原样搬到稳态段上:

  steady_suff_len   仅包长幅度: absdir + nobytes + nointervals   (= e6_suff_mag_only)
  steady_suff_dir   仅方向:     pad + nobytes + nointervals      (= e6_suff_dir_only)
  steady_suff_time  仅时间:     nosizes + nobytes                (= bi_abl_no_sizes_nobytes)

变换定义与 504 流版本逐字相同 (pipeline/pcap_split/simulate_obfuscation_bi.py 的 generate),
施加于全部 split, 每折从预训练 ckpt 重新微调。折 = data/processed/e3_folds/steady/fold{k}
(run_e6_steady.py 建的 448 流分配, 不重造), 所以与 steady 全特征条件逐条配对。
NetMamba+ 走 acc_loss 选模规则 (MASQUE_SELRULE=acc_loss, 本脚本强制); RF 与
rf_on_icoin_r2_folds.py 同输入同参数 (300 trees, random_state=0, train+valid 训练)。

WSL MAMBA 环境:
    python experiments/run_steady_suff.py --stage all
输出 outputs/e6_steady_sel/summary_steady_suff.json
"""

import argparse
import json
import os
import sys

os.environ["MASQUE_SELRULE"] = "acc_loss"

_ROOT = os.environ.get("MASQUE_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "experiments"))
sys.path.insert(0, os.path.join(_ROOT, "pipeline", "pcap_split"))

import run_kfold_e3 as _e3  # noqa: E402
import selrule_util  # noqa: E402
from run_kfold_e3 import load_oof, summarize, train_fold  # noqa: E402
from simulate_obfuscation_bi import generate  # noqa: E402

FOLD_ROOT = os.path.join(_ROOT, "data", "processed", "e3_folds")
SRC = "steady"
N_SPLITS = 5
OUT_ROOT = selrule_util.out_dir(os.path.join(_ROOT, "outputs", "e3_kfold"))
_e3.OUT_ROOT = OUT_ROOT
SUMMARY = os.path.join(_ROOT, "outputs", "e6_steady_sel", "summary_steady_suff.json")

CONDS = [
    ("steady_suff_len",  ["absdir", "nobytes", "nointervals"], "steady length only"),
    ("steady_suff_dir",  ["pad", "nobytes", "nointervals"],    "steady direction only"),
    ("steady_suff_time", ["nosizes", "nobytes"],               "steady timing only"),
]


def stage_data():
    for cid, tf, note in CONDS:
        for k in range(N_SPLITS):
            src = os.path.join(FOLD_ROOT, SRC, f"fold{k}")
            dst = os.path.join(FOLD_ROOT, cid, f"fold{k}")
            assert os.path.exists(os.path.join(src, "data-test.json")), src
            generate(src, dst, tf, "both", ["train", "valid", "test"], False)


def stage_train(epochs):
    todo = [(cid, k) for cid, _, _ in CONDS for k in range(N_SPLITS)]
    for i, (cid, k) in enumerate(todo, 1):
        d = os.path.join(OUT_ROOT, cid, f"fold{k}")
        if all(os.path.exists(os.path.join(d, p)) for p in selrule_util.pred_files()):
            print(f"[{i}/{len(todo)}] 跳过已完成 {cid} fold{k}")
            continue
        print(f"[{i}/{len(todo)}] 训练 {cid} fold{k}", flush=True)
        train_fold(cid, k, 0, epochs)
        if not selrule_util.dump_pair(os.path.join(FOLD_ROOT, cid, f"fold{k}"), d, 0):
            sys.exit(f"!!! dump 失败: {cid} fold{k}")
        for name in selrule_util.all_ckpts():
            if name != selrule_util.ckpt_name():
                os.remove(os.path.join(d, name))


def rf_cond(cid):
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from rf_on_icoin_r2_folds import feats, wilson

    cm = [[0, 0], [0, 0]]
    per_fold = []
    for k in range(N_SPLITS):
        d = os.path.join(FOLD_ROOT, cid, f"fold{k}")
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
    n = sum(map(sum, cm)); ok = cm[0][0] + cm[1][1]
    return {"acc": ok / n, "ci95": wilson(ok, n), "cm": cm, "per_fold": per_fold}


def stage_summary():
    from scipy.stats import binomtest

    ref, _ = load_oof(SRC, 0)          # steady 全特征 (e3_kfold_sel/steady)
    out = {"source_folds": os.path.join("data", "processed", "e3_folds", SRC),
           "selection_rule": selrule_util.RULE_DESC, "conditions": {}}
    for cid, tf, note in CONDS:
        oof, per_fold = load_oof(cid, 0)
        if oof is None:
            print(f"(未完成: {cid})")
            continue
        s = summarize(oof)
        s["note"], s["transforms"] = note, tf
        s["fold_accs"] = [f["acc"] for f in per_fold]
        keys = sorted(set(oof) & set(ref))
        b = sum(1 for x in keys if ref[x]["pred"] == ref[x]["label"] and oof[x]["pred"] != oof[x]["label"])
        c = sum(1 for x in keys if ref[x]["pred"] != ref[x]["label"] and oof[x]["pred"] == oof[x]["label"])
        s["mcnemar_vs_steady_full"] = {"n": len(keys), "b_full_only": b, "c_cond_only": c,
                                       "p": binomtest(b, b + c, 0.5).pvalue if b + c else 1.0}
        s["rf"] = rf_cond(cid)
        out["conditions"][cid] = s
        print(f"{note:<24} NetMamba+ {s['acc']*100:6.2f}% [{s['ci95'][0]*100:.2f}, "
              f"{s['ci95'][1]*100:.2f}] cm {s['cm']}  vs full b/c {b}/{c}   "
              f"RF {s['rf']['acc']*100:6.2f}% cm {s['rf']['cm']}", flush=True)
    labels = [v["label"] for v in ref.values()]
    out["majority_baseline"] = max(labels.count(0), labels.count(1)) / len(labels)
    out["caveat"] = "448 流 (>= 100 包) 子集, 多数类 57.81%; 与 504 流的 Table I 数字不可直接相减。"
    os.makedirs(os.path.dirname(SUMMARY), exist_ok=True)
    with open(SUMMARY, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"多数类基线 {out['majority_baseline']*100:.2f}%\n已写入 {SUMMARY}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "data", "train", "summary"])
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    if args.stage in ("all", "data"):
        stage_data()
    if args.stage in ("all", "train"):
        stage_train(args.epochs)
    if args.stage in ("all", "summary"):
        stage_summary()


if __name__ == "__main__":
    main()
