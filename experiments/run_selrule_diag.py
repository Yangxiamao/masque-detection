"""[2026-10-12 ICOIN r2 §7.5] 选模规则诊断: 改标后 Full (97.02%) 偏低是不是选模伪影?

现象: 改标后 Full 低于多数消融条件, 15 个错误全是 Direct -> MASQUE, ROC-AUC 却是 1.0000;
选中的 epoch 原标签 10/17/10/3/4, 改标 3/3/3/2/3。候选解释: 验证集 (41 条) 过早到 100%,
"acc 第一次最高"的规则选中了欠训练模型。

做法: 两套标签的 Full 各重跑 5 折, 超参与 run_kfold_e3.py 逐字相同, 由 finetune_selrule.py
在同一次训练里按 4 种规则各存一个 checkpoint (见该文件), 然后逐条 dump 测试集预测。

预先指定 (看结果之前): 主规则 = acc_loss (acc 最高, 打平比 valid loss), 主比较 =
同一标签下 acc_loss vs acc_first 的 exact McNemar, 两套标签 2 个比较做 Holm。
min_loss / last 仅作补充, 不进检验族。

一致性检查: acc_first 的预测必须与 outputs/e3{,r}_kfold/full 里已有的 preds.json 逐条相同,
否则说明训练不可复现, 本诊断结论要打折扣。

    python experiments/run_selrule_diag.py            # 训练 + dump + 汇总
    python experiments/run_selrule_diag.py --summary  # 只汇总
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time

_ROOT = os.environ.get("MASQUE_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "experiments"))

from run_kfold_e3 import (BATCH_SIZE, BYTE_PACKET_NUM, EPOCHS, HEADER_BYTE_NUM,  # noqa: E402
                          INTERVAL_ENCODING, N_SPLITS, PAYLOAD_BYTE_NUM, SEQ_PACKET_NUM,
                          STRIDE_LEN, holm, mcnemar, wilson_ci)

OUT_ROOT = os.path.join(_ROOT, "outputs", "selrule_diag")
LABELS = {
    "orig":    {"folds": os.path.join(_ROOT, "data", "processed", "e3_folds", "full"),
                "ref":   os.path.join(_ROOT, "outputs", "e3_kfold", "full")},
    "relabel": {"folds": os.path.join(_ROOT, "data", "processed", "e3r_folds", "full"),
                "ref":   os.path.join(_ROOT, "outputs", "e3r_kfold", "full")},
}
RULES = {"acc_first": "checkpoint-best.pth", "acc_loss": "ckpt-acc_loss.pth",
         "min_loss": "ckpt-min_loss.pth", "last": "ckpt-last.pth"}
PRIMARY = "acc_loss"
NOISE_JSON = os.path.join(_ROOT, "outputs", "e3_kfold", "error_analysis_e3.json")


def fold_out(lab, k):
    return os.path.join(OUT_ROOT, lab, f"fold{k}")


def fold_done(lab, k):
    d = fold_out(lab, k)
    return os.path.exists(os.path.join(d, "selrule.json")) and all(
        os.path.exists(os.path.join(d, f"preds_{r}.json")) for r in RULES)


def train_fold(lab, k, epochs):
    data_dir = os.path.join(LABELS[lab]["folds"], f"fold{k}")
    out_dir = fold_out(lab, k)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)   # log.txt 是追加写入, 重跑前必须清空
    os.makedirs(out_dir)
    with open(os.path.join(data_dir, "metadata.json"), encoding="utf-8") as f:
        num_class = len(json.load(f)["name_to_idx"])
    ckpt_path = os.path.join(_ROOT, "NetMambaPlus", "dataset", "checkpoints",
                             "pre-trained checkpoints", "fuse3_mamba.pth")
    # 与 run_kfold_e3.train_fold 逐字相同的参数, 只把入口换成透传包装器
    cmd = [sys.executable, "-u", os.path.join(_ROOT, "experiments", "finetune_selrule.py"),
           "--num_packet", str(BYTE_PACKET_NUM),
           "--num_packet_byte", str(HEADER_BYTE_NUM + PAYLOAD_BYTE_NUM),
           "--seq_len", str(SEQ_PACKET_NUM),
           "--stride_size", str(STRIDE_LEN),
           "--batch_size", str(BATCH_SIZE),
           "--epochs", str(epochs),
           "--seed", "0",
           "--nb_classes", str(num_class),
           "--finetune", ckpt_path,
           "--data_path", data_dir,
           "--output_dir", out_dir,
           "--log_dir", out_dir,
           "--model", "fuse3_mamba_classifier",
           "--dataset_type", "byte_size_interval",
           "--size_key", "signed_sizes",
           "--interval_encoding", INTERVAL_ENCODING,
           "--no_amp"]
    print(f"\n{'='*74}\nSTEP train {lab} fold{k}  {time.strftime('%H:%M:%S')}\n{'='*74}", flush=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, cwd=os.path.join(_ROOT, "NetMambaPlus", "src", "scripts"))
    if proc.returncode != 0:
        sys.exit(f"!!! 训练失败: {lab} fold{k}")
    print(f"--- {lab} fold{k} 训练用时 {time.time()-t0:.1f}s", flush=True)


def dump_fold(lab, k):
    from eval_dump_preds import dump
    data_dir = os.path.join(LABELS[lab]["folds"], f"fold{k}")
    out_dir = fold_out(lab, k)
    for rule, ck in RULES.items():
        acc = dump(data_dir, os.path.join(out_dir, ck), os.path.join(out_dir, f"preds_{rule}.json"))
        if rule == "acc_first":
            with open(os.path.join(out_dir, "test_stats.json"), encoding="utf-8") as f:
                ref = json.load(f)["acc"]
            if abs(acc - ref) > 1e-9:
                sys.exit(f"!!! dump acc {acc} != test_stats {ref}: {lab} fold{k}")
    for ck in RULES.values():
        os.remove(os.path.join(out_dir, ck))   # 4 x 22.5 MB/折, 预测已落盘


# --------------------------------------------------------------------------- #

def load_oof(lab, rule):
    oof = {}
    for k in range(N_SPLITS):
        with open(os.path.join(fold_out(lab, k), f"preds_{rule}.json"), encoding="utf-8") as f:
            oof.update(json.load(f))
    return oof


def load_ref(lab):
    oof = {}
    for k in range(N_SPLITS):
        with open(os.path.join(LABELS[lab]["ref"], f"fold{k}", "preds.json"), encoding="utf-8") as f:
            oof.update(json.load(f))
    return oof


def stats(oof, noise):
    from sklearn.metrics import roc_auc_score
    keys = sorted(oof)
    n = len(keys)
    correct = sum(oof[x]["pred"] == oof[x]["label"] for x in keys)
    cm = [[0, 0], [0, 0]]
    for x in keys:
        cm[oof[x]["label"]][oof[x]["pred"]] += 1
    lo, hi = wilson_ci(correct, n)
    clean = [x for x in keys if x not in noise]
    clean_ok = sum(oof[x]["pred"] == oof[x]["label"] for x in clean)
    d_probs = [oof[x]["prob1"] for x in keys if oof[x]["label"] == 0]
    return {"n": n, "correct": correct, "acc": correct / n, "ci95": [lo, hi], "cm": cm,
            "auc": roc_auc_score([oof[x]["label"] for x in keys], [oof[x]["prob1"] for x in keys]),
            "acc_on_493_uncontested": clean_ok / len(clean), "n_uncontested": len(clean),
            "direct_prob1_median": sorted(d_probs)[len(d_probs) // 2],
            "direct_prob1_max": max(d_probs)}


def summarize():
    with open(NOISE_JSON, encoding="utf-8") as f:
        noise = set(json.load(f)["label_noise_flows"])
    out = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "primary_rule": PRIMARY,
           "rules": {"acc_first": "valid acc 第一次最高 (fine-tune.py 原规则)",
                     "acc_loss": "valid acc 最高, 打平取 valid loss 最低 (预先指定的主规则)",
                     "min_loss": "valid loss 最低 (补充)", "last": "最后一个 epoch (补充)"},
           "labels": {}}
    primary_tests = []
    for lab in LABELS:
        sel = {r: [] for r in RULES}
        for k in range(N_SPLITS):
            with open(os.path.join(fold_out(lab, k), "selrule.json"), encoding="utf-8") as f:
                s = json.load(f)["selected_epoch"]
            for r in RULES:
                sel[r].append(s[r])
        ref = load_ref(lab)
        first = load_oof(lab, "acc_first")
        repro = {"pred_mismatch": sum(first[x]["pred"] != ref[x]["pred"] for x in ref),
                 "max_abs_prob1_diff": max(abs(first[x]["prob1"] - ref[x]["prob1"]) for x in ref),
                 "n": len(ref)}
        res = {"selected_epoch": sel, "repro_vs_existing_preds": repro, "rules": {}}
        for r in RULES:
            oof = load_oof(lab, r)
            res["rules"][r] = stats(oof, noise)   # 493 条无争议流上两套标签完全相同
            if r != "acc_first":
                t = mcnemar(first, oof)   # b = 只有 acc_first 对, c = 只有该规则对
                res["rules"][r]["vs_acc_first"] = t
                if r == PRIMARY:
                    primary_tests.append((lab, t))
        out["labels"][lab] = res
    for (lab, t), adj in zip(primary_tests, holm([t["p"] for _, t in primary_tests])):
        out["labels"][lab]["rules"][PRIMARY]["vs_acc_first"]["p_holm_family2"] = adj
    os.makedirs(OUT_ROOT, exist_ok=True)
    path = os.path.join(OUT_ROOT, "summary_selrule.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    for lab, res in out["labels"].items():
        rp = res["repro_vs_existing_preds"]
        print(f"\n=== {lab}   复现核对: 预测不一致 {rp['pred_mismatch']}/{rp['n']}, "
              f"prob1 最大差 {rp['max_abs_prob1_diff']:.2e}")
        print(f"{'rule':<10}{'epochs':<20}{'OOF':>8}{'95% CI':>18}{'AUC':>8}{'493':>8}  cm   vs acc_first b/c p")
        for r, s in res["rules"].items():
            t = s.get("vs_acc_first")
            tt = f"{t['b_only_A_correct']}/{t['c_only_B_correct']} p={t['p']:.2g}" if t else ""
            if t and "p_holm_family2" in t:
                tt += f" holm={t['p_holm_family2']:.2g}"
            print(f"{r:<10}{str(res['selected_epoch'][r]):<20}{s['acc']*100:>8.2f}"
                  f"{'[%.2f, %.2f]' % (s['ci95'][0]*100, s['ci95'][1]*100):>18}{s['auc']:>8.4f}"
                  f"{s['acc_on_493_uncontested']*100:>8.2f}  {s['cm']}  {tt}")
    print(f"\n已写入 {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--labels", default="relabel,orig")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if not args.summary:
        for lab in args.labels.split(","):
            for k in range(N_SPLITS):
                if fold_done(lab, k) and not args.force:
                    print(f"跳过已完成: {lab} fold{k}")
                    continue
                train_fold(lab, k, args.epochs)
                dump_fold(lab, k)
        print(f"SELRULE_TRAIN_END {time.strftime('%H:%M:%S')}", flush=True)
    summarize()


if __name__ == "__main__":
    main()
