# -*- coding: utf-8 -*-
"""[2026-09-08] E6-C: 稳态段 vs 握手段 —— 分类器学的是"隧道"还是"新建连接"?

## 要排除的解释

主管道取每条流的【前 50 个包】。这一段里同时住着三样东西:
  * 采集程序在每个 MASQUE 域名前起隧道、访问后拆隧道 (Direct 侧无对应操作),
    于是每条 MASQUE 流都带一次全新隧道建立, 每条 Direct 流都不带;
  * Retry 地址验证: 259/270 条 MASQUE 流有, 234 条 Direct 流全无 (E1 的 P-C);
  * 真正想测的东西: 隧道运行期的包长/方向/时序结构。

于是"分类器只是在区分冷启动连接与温连接"是一个论文没有排除的解释。C123 消融
删掉了 Retry 帧, 但删不掉它引入的那个往返, 论文自己也写了 "C3 removes the Retry
frame but not the round trip it induces"。

## 设计

对包数 >= 100 的流取【第 51-100 个包】—— 完全处于稳态传输阶段, 不含任何握手。
成对的对照是【同一批流的前 50 个包】: 若只比 504 条流的 98.02%, 就分不清精度变化
来自"换了流的哪一段"还是"换了哪些流"(>= 100 包的过滤把 504 条筛成了 448 条)。

两个条件的流集合与文件名逐条相同 -> 复用同一份折分配 -> McNemar 可用。

    python experiments/run_e6_steady.py --stage all
"""

import argparse
import json
import math
import os
import subprocess
import sys

_ROOT = os.environ.get("MASQUE_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "experiments"))

import run_kfold_e3 as _e3  # noqa: E402
import selrule_util  # noqa: E402
from run_kfold_e3 import load_oof, summarize, train_fold, wilson_ci  # noqa: E402

PROC = os.path.join(_ROOT, "data", "processed")
OUT_ROOT = os.path.join(_ROOT, "outputs", "e3_kfold")     # 与其他条件同根, 复用 load_oof
FOLD_ROOT = os.path.join(PROC, "e3_folds")
ASSIGNMENT = os.path.join(_ROOT, "outputs", "e6_steady", "fold_assignment_448.json")
SUMMARY = os.path.join(_ROOT, "outputs", "e6_steady", "summary_e6_steady.json")
N_SPLITS = 5

# [2026-10-13 ICOIN r2 §7.5] MASQUE_SELRULE=acc_loss -> 新选模规则: 训练/汇总写到 e3_kfold_sel 与
# e6_steady_sel, 折数据与折分配照旧 (不重造)。11 条改标流都不足 100 包, 不在这 448 条里, 标签不受影响。
# 未设置时恒等。
OLD_OUT_ROOT = OUT_ROOT
if selrule_util.SELRULE:
    OUT_ROOT = selrule_util.out_dir(OUT_ROOT)
    _e3.OUT_ROOT = OUT_ROOT   # train_fold / load_oof 读的是 run_kfold_e3 的模块全局
    SUMMARY = os.path.join(selrule_util.out_dir(os.path.dirname(SUMMARY)), os.path.basename(SUMMARY))

# (条件 id, 切流目录, 特征目录, 说明)
CONDS = [
    ("steady", "quic_vs_masque_bi_seg50_50", "e6_steady_feat",  "稳态段: 第 51-100 包"),
    ("head448", "quic_vs_masque_bi_head448", "e6_head448_feat", "对照: 同一批流的前 50 包"),
]


def sh(cmd, **kw):
    p = subprocess.run(cmd, **kw)
    if p.returncode != 0:
        sys.exit(f"命令失败: {cmd}")


def stage_data():
    for cid, pcap_dir, feat_dir, note in CONDS:
        src = os.path.join(PROC, pcap_dir)
        if not os.path.isdir(src):
            sys.exit(f"切流目录不存在: {src}  (先跑 split_quic_vs_masque_bi_segment.py)")
        n = sum(len(os.listdir(os.path.join(src, d))) for d in os.listdir(src))
        print(f"\n=== {cid} ({note}): {n} 条流 ===")

        if not os.path.exists(os.path.join(PROC, feat_dir, "data.json")):
            sh([sys.executable, os.path.join(_ROOT, "pipeline", "feat_extract",
                                             "feat_quic_vs_masque_bi.py"),
                "--pcap_dir", pcap_dir, "--save_dir", feat_dir])
        if not os.path.exists(os.path.join(PROC, feat_dir, "data-test.json")):
            sh([sys.executable, os.path.join(_ROOT, "pipeline", "split_data.py"),
                "--feat_dir", os.path.join(PROC, feat_dir)])

    # 两个条件的样本集合必须逐条相同, 否则配对检验无效
    keys = {}
    for cid, _, feat_dir, _ in CONDS:
        s = set()
        for sp in ["train", "valid", "test"]:
            with open(os.path.join(PROC, feat_dir, f"data-{sp}.json"), encoding="utf-8") as f:
                s |= {os.path.basename(it["pcap_file"]) for it in json.load(f)}
        keys[cid] = s
    a, b = keys["steady"], keys["head448"]
    print(f"\n样本集合: steady {len(a)} 条, head448 {len(b)} 条, 交集 {len(a & b)} 条")
    if a != b:
        print(f"⚠️ 两者不一致 (仅 steady {len(a-b)}, 仅 head448 {len(b-a)});"
              f" 将只在交集上做配对比较")

    # 折分配: 流集合与 E3 的 504 条不同, 必须另建一份
    os.makedirs(os.path.dirname(ASSIGNMENT), exist_ok=True)
    for cid, _, feat_dir, _ in CONDS:
        sh([sys.executable, os.path.join(_ROOT, "pipeline", "split_data_kfold.py"),
            "--feat_dir", os.path.join(PROC, feat_dir),
            "--out_root", os.path.join(FOLD_ROOT, cid),
            "--assignment", ASSIGNMENT])


def stage_train(epochs):
    todo = [(cid, k) for cid, _, _, _ in CONDS for k in range(N_SPLITS)]
    for i, (cid, k) in enumerate(todo, 1):
        d = os.path.join(OUT_ROOT, cid, f"fold{k}")
        if all(os.path.exists(os.path.join(d, p)) for p in selrule_util.pred_files()):
            print(f"[{i}/{len(todo)}] 跳过已完成 {cid} fold{k}")
            continue
        print(f"[{i}/{len(todo)}] 训练 {cid} fold{k}")
        train_fold(cid, k, 0, epochs)
        if selrule_util.SELRULE:
            # 原规则 ckpt -> preds_acc_first.json (核对 test_stats), acc_loss -> preds.json; 只留主 ckpt
            if not selrule_util.dump_pair(os.path.join(FOLD_ROOT, cid, f"fold{k}"), d, 0):
                sys.exit(f"!!! dump 失败: {cid} fold{k}")
            for name in selrule_util.all_ckpts():
                if name != selrule_util.ckpt_name():
                    os.remove(os.path.join(d, name))
            continue
        sh([sys.executable, os.path.join(_ROOT, "experiments", "eval_dump_preds.py"),
            "--data_dir", os.path.join(FOLD_ROOT, cid, f"fold{k}"),
            "--ckpt", os.path.join(d, "checkpoint-best.pth"),
            "--out", os.path.join(d, "preds.json"),
            "--verify", os.path.join(d, "test_stats.json")])


def stage_summary():
    from scipy.stats import binomtest

    res = {}
    for cid, _, _, note in CONDS:
        oof, per_fold = load_oof(cid, 0)
        if oof is None:
            print(f"(未完成: {cid})")
            continue
        s = summarize(oof)
        s["note"] = note
        s["fold_accs"] = [f["acc"] for f in per_fold]
        res[cid] = s

    print(f"\n{'='*92}")
    print("E6-C  稳态段 vs 握手段 (同一批 >= 100 包的流, 5 折 OOF)")
    print(f"{'='*92}")
    print(f"{'条件':<28}{'n':>6}{'OOF Acc':>10}{'95% CI':>20}{'错':>5}  混淆矩阵")
    print("-" * 92)
    for cid, s in res.items():
        ci = f"[{s['ci95'][0]*100:.2f}, {s['ci95'][1]*100:.2f}]"
        print(f"{s['note']:<26}{s['n']:>6}{s['acc']*100:>9.2f}%{ci:>20}"
              f"{s['n']-s['correct']:>5}  {s['cm']}")
    print("-" * 92)

    out = {"conditions": res}
    if len(res) == 2:
        oa, _ = load_oof("steady", 0)
        ob, _ = load_oof("head448", 0)
        keys = sorted(set(oa) & set(ob))
        b = sum(1 for k in keys if oa[k]["pred"] == oa[k]["label"] and ob[k]["pred"] != ob[k]["label"])
        c = sum(1 for k in keys if oa[k]["pred"] != oa[k]["label"] and ob[k]["pred"] == ob[k]["label"])
        p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
        out["mcnemar_steady_vs_head"] = {"n": len(keys), "b_steady_only": b,
                                         "c_head_only": c, "p": p}
        print(f"\n配对 McNemar (稳态 vs 头部, n={len(keys)}): "
              f"仅稳态对 {b}, 仅头部对 {c}, p = {p:.4f}")
        maj = max(sum(1 for v in oa.values() if v["label"] == L) for L in (0, 1)) / len(oa)
        print(f"该子集的多数类基线: {maj*100:.2f}%")
        out["majority_baseline"] = maj

    out["caveat"] = ("稳态段与头部段用同一批流、同一份折分配, 故配对; 但流集合 (448) "
                     "与主实验 (504) 不同, 不能把稳态段的数字直接与 98.02% 相减。")
    if selrule_util.SELRULE:
        # 旧规则对照 + 复现核对: 同一次训练里原规则 ckpt 的预测 vs 原目录 (e3_kfold/<cid>) 的预测
        out["selection_rule"] = selrule_util.RULE_DESC
        out["old_rule"], out["repro_acc_first_vs_old"], out["selected_epochs"] = {}, {}, {}
        for cid, _, _, _ in CONDS:
            old, first, sel = {}, {}, {"acc_first": [], "acc_loss": []}
            for k in range(N_SPLITS):
                with open(os.path.join(OLD_OUT_ROOT, cid, f"fold{k}", "preds.json"), encoding="utf-8") as f:
                    old.update(json.load(f))
                with open(os.path.join(OUT_ROOT, cid, f"fold{k}", "preds_acc_first.json"), encoding="utf-8") as f:
                    first.update(json.load(f))
                with open(os.path.join(OUT_ROOT, cid, f"fold{k}", "selrule.json"), encoding="utf-8") as f:
                    s = json.load(f)["selected_epoch"]
                sel["acc_first"].append(s["acc_first"])
                sel["acc_loss"].append(s["acc_loss"])
            out["old_rule"][cid] = summarize(old)
            out["repro_acc_first_vs_old"][cid] = {
                "n": len(old), "pred_mismatch": sum(first[x]["pred"] != old[x]["pred"] for x in old),
                "max_abs_prob1_diff": max(abs(first[x]["prob1"] - old[x]["prob1"]) for x in old)}
            out["selected_epochs"][cid] = sel
            print(f"  [旧规则] {cid}: {out['old_rule'][cid]['acc']*100:.2f}%  复现核对 "
                  f"{out['repro_acc_first_vs_old'][cid]}  epochs {sel}")
    os.makedirs(os.path.dirname(SUMMARY), exist_ok=True)
    with open(SUMMARY, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n已写入 {SUMMARY}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "data", "train", "summary"])
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    if args.stage in ("all", "data"):
        have = all(os.path.exists(os.path.join(FOLD_ROOT, c[0], f"fold{k}", "data-test.json"))
                   for c in CONDS for k in range(N_SPLITS))
        if selrule_util.SELRULE and have:
            print(f"[sel] 复用已有折 {FOLD_ROOT}/{{steady,head448}}, 不重造")
        else:
            stage_data()
    if args.stage in ("all", "train"):
        stage_train(args.epochs)
    if args.stage in ("all", "summary"):
        stage_summary()


if __name__ == "__main__":
    main()
