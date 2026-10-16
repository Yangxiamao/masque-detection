"""[2026-10-12] ICOIN r2: 教授第 2 轮意见 #1 (T3-dir 直接对抗训练) / #2 (表 I No-Length)。

见 docs/01_Planning/Handoff_ICOIN_Revision_2026-10-12.md §3 #1 #2。

每个条件 = 对每折的 train / valid / test【全部】施加同一变换链, 从公开预训练 ckpt
重新微调, 在同样变换过的 test 上评估 —— 即"知道防御的攻击者在防御后的流量上重训"。
折数据集直接由基底折 (e3_folds/full 或 e3r_folds/full) 逐条变换得到, 所以折分配、
valid 划分与 E3 逐字相同, 与 E3 / E6 的所有条件可做配对 McNemar。

    python experiments/run_icoin_r2.py --labels orig                 # 原标签, 全部条件
    python experiments/run_icoin_r2.py --labels relabel              # 改标 (需先 make_relabel_folds.py)
    python experiments/run_icoin_r2.py --labels orig --only t3dir_san
    python experiments/run_icoin_r2.py --labels orig --summary

数据: data/processed/icoin_r2_folds/<labels>/<cond>/fold{k}/
输出: outputs/icoin_r2/<labels>/<cond>/fold{k}/ ; 汇总 outputs/icoin_r2/<labels>/summary.json
超参与 run_kfold_e3.py 逐字相同 (直接调用其 train_fold / dump_fold)。
"""

import argparse
import copy
import json
import os
import shutil
import sys
import time

_ROOT = os.environ.get("MASQUE_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "experiments"))
sys.path.insert(0, os.path.join(_ROOT, "pipeline", "pcap_split"))

import run_kfold_e3 as e3                                  # noqa: E402
import selrule_util                                        # noqa: E402
from simulate_obfuscation_bi import TRANSFORMS as _SIM_TF  # noqa: E402
from sanitize_bytes import (tf_sanitize_bytes, tf_sanitize_hdrnorm,  # noqa: E402
                            tf_sanitize_hdrnorm_types)

TRANSFORMS = dict(_SIM_TF)
TRANSFORMS["sanitize"] = tf_sanitize_bytes
TRANSFORMS["hdrnorm"] = tf_sanitize_hdrnorm
TRANSFORMS["hdrnorm_types"] = tf_sanitize_hdrnorm_types

PROC = os.path.join(_ROOT, "data", "processed")
BASE_FOLDS = {"orig": os.path.join(PROC, "e3_folds", "full"),
              "relabel": os.path.join(PROC, "e3r_folds", "full")}
# 全特征基线的 OOF (配对比较用); MASQUE_SELRULE=acc_loss 时指向同规则的 *_sel
BASE_OUT = {"orig": selrule_util.out_dir(os.path.join(_ROOT, "outputs", "e3_kfold")),
            "relabel": selrule_util.out_dir(os.path.join(_ROOT, "outputs", "e3r_kfold"))}
N_SPLITS = 5
SPLITS = ["train", "valid", "test"]

# (条件 id, 变换链, 说明)
CONDITIONS = [
    # ---- #1 T3-dir 直接对抗训练 ----
    ("t3dir_raw",    ["pad", "shape", "absdir"],             "T3-dir raw: 字节通道原样"),
    ("t3dir_san",    ["pad", "shape", "absdir", "sanitize"], "T3-dir sanitized: IP/UDP 头钉为模板, Length 清零, 补零区随机"),
    ("t3dir_strict", ["pad", "shape", "absdir", "nobytes"],  "T3-dir strict: 字节全零 (输入完全相同 -> trivial 下界)"),
    # ---- #2 表 I No-Length (|size| 恒为 1350, 保号) ----
    ("nolen_pad",         ["pad"],             "No-Length: 仅 pad (字节通道仍含长度)"),
    ("nolen_pad_nobytes", ["pad", "nobytes"],  "No-Length strict: pad + 字节全零 (主数字)"),
    ("nolen_pad_san",     ["pad", "sanitize"], "No-Length sanitized: pad + sanitize"),
    # ---- #1 选项 C: 部署常量对齐后再关三条侧信道 (基底 = C123 折, C3 已从 pcap 删掉 Retry) ----
    ("t3dir_c123_san",     ["pad", "shape", "absdir", "sanitize"],      "C0: C123 基底 + T3-dir sanitized (只删 Retry 的对照)"),
    ("t3dir_c123_hdr",     ["pad", "shape", "absdir", "hdrnorm"],       "C1: + CID 长度 8/0、token 去掉, 包类型保留"),
    ("t3dir_c123_hdrtype", ["pad", "shape", "absdir", "hdrnorm_types"], "C2: + 包类型也统一 (近平凡下界)"),
]
# 条件 -> 基底折 (默认 full)
COND_BASE = {"t3dir_c123_san": "c123", "t3dir_c123_hdr": "c123", "t3dir_c123_hdrtype": "c123"}
COND_BY_ID = {c[0]: c for c in CONDITIONS}


def fold_root(labels):
    return os.path.join(PROC, "icoin_r2_folds", labels)


def out_root(labels):
    return selrule_util.out_dir(os.path.join(_ROOT, "outputs", "icoin_r2", labels))


def build_folds(labels, cid):
    tf_names = COND_BY_ID[cid][1]
    base = os.path.join(os.path.dirname(BASE_FOLDS[labels]), COND_BASE.get(cid, "full"))
    assert os.path.isdir(base), f"基底折不存在: {base}"
    tag = "+".join(tf_names)
    for k in range(N_SPLITS):
        dst = os.path.join(fold_root(labels), cid, f"fold{k}")
        if all(os.path.exists(os.path.join(dst, f"data-{s}.json")) for s in SPLITS):
            continue
        os.makedirs(dst, exist_ok=True)
        for s in SPLITS:
            with open(os.path.join(base, f"fold{k}", f"data-{s}.json"), encoding="utf-8") as f:
                items = json.load(f)
            out = []
            for it in items:
                new = it
                for t in tf_names:
                    new = TRANSFORMS[t](new)
                new = copy.deepcopy(new) if new is it else new
                new["variant"] = tag
                out.append(new)
            with open(os.path.join(dst, f"data-{s}.json"), "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
        shutil.copy(os.path.join(base, f"fold{k}", "metadata.json"),
                    os.path.join(dst, "metadata.json"))
        print(f"[prepare] {labels}/{cid}/fold{k}  tf={tag}", flush=True)


def majority_oof(oof):
    """多数类预测器 (全判 MASQUE=1) 的伪 OOF, 用于与 majority 的配对检验。"""
    return {k: {"pred": 1, "label": v["label"]} for k, v in oof.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, choices=list(BASE_FOLDS))
    ap.add_argument("--only", default=None)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    conds = ([COND_BY_ID[c.strip()] for c in args.only.split(",")] if args.only else CONDITIONS)

    # 让 e3 的 train_fold / dump_fold / load_oof 指向本实验的目录
    e3.FOLD_ROOT = fold_root(args.labels)
    e3.OUT_ROOT = out_root(args.labels)

    if not args.summary:
        t_all = time.time()
        for cid in (c[0] for c in conds):
            build_folds(args.labels, cid)
        total, done = len(conds) * N_SPLITS, 0
        for cid, _, note in conds:
            for k in range(N_SPLITS):
                done += 1
                od = os.path.join(e3.OUT_ROOT, cid, f"fold{k}")
                if (all(os.path.exists(os.path.join(od, p)) for p in selrule_util.pred_files())
                        and not args.force):
                    print(f"[{done}/{total}] 跳过已完成: {cid} fold{k}", flush=True)
                    continue
                # fine-tune.py 的 log.txt 是追加写入: 重跑前清空该折目录 (handoff §6-4)
                if os.path.isdir(od):
                    shutil.rmtree(od)
                print(f"[{done}/{total}] STEP train {args.labels}/{cid} fold{k}  ({note})", flush=True)
                e3.train_fold(cid, k, 0, e3.EPOCHS)
                if not e3.dump_fold(cid, k, 0):
                    print(f"!!! FAIL dump {cid} fold{k}", flush=True)
                e3.drop_ckpt(cid, k, 0, keep=False)
        print(f"\n全部训练用时 {time.time()-t_all:.1f}s", flush=True)

    # ---------------- 汇总 ----------------
    results, oofs = {}, {}
    for cid, tfs, note in CONDITIONS:
        oof, per_fold = e3.load_oof(cid, 0)
        if oof is None:
            continue
        s = e3.summarize(oof)
        s["per_fold"] = per_fold
        s["transforms"] = tfs
        s["note"] = note
        results[cid] = s
        oofs[cid] = oof

    # 全特征基线 OOF (配对用)
    e3_out_saved = e3.OUT_ROOT
    e3.OUT_ROOT = BASE_OUT[args.labels]
    full_oof, _ = e3.load_oof("full", 0)
    e3.OUT_ROOT = e3_out_saved

    def test(a_name, a, b_name, b):
        r = e3.mcnemar(a, b)
        r["A"], r["B"] = a_name, b_name
        return r

    families = {}
    # 族 1 (#1): T3-dir 三变体各与 majority 比较 + 变体两两比较
    t3 = [c for c in ("t3dir_raw", "t3dir_san", "t3dir_strict") if c in oofs]
    fam = []
    for c in t3:
        fam.append(test(c, oofs[c], "majority", majority_oof(oofs[c])))
    for i in range(len(t3)):
        for j in range(i + 1, len(t3)):
            fam.append(test(t3[i], oofs[t3[i]], t3[j], oofs[t3[j]]))
    families["t3dir"] = fam
    # 族 2 (#2): No-Length 各变体与全特征基线比较 (并入表 I 的 ablation 族时需重算 Holm)
    fam = []
    if full_oof is not None:
        for c in ("nolen_pad", "nolen_pad_nobytes", "nolen_pad_san"):
            if c in oofs:
                fam.append(test("full", full_oof, c, oofs[c]))
    families["nolen_vs_full"] = fam
    # 族 3 (#1 选项 C): 常量对齐系列各与 majority 比较, 并与 t3dir_san 比较
    fam = []
    for c in ("t3dir_c123_san", "t3dir_c123_hdr", "t3dir_c123_hdrtype"):
        if c in oofs:
            fam.append(test(c, oofs[c], "majority", majority_oof(oofs[c])))
            if "t3dir_san" in oofs:
                fam.append(test("t3dir_san", oofs["t3dir_san"], c, oofs[c]))
    families["c123_hdrnorm"] = fam
    for fam in families.values():
        for r, adj in zip(fam, e3.holm([r["p"] for r in fam])):
            r["p_holm_within_family"] = adj

    n = next(iter(results.values()))["n"] if results else 0
    n_masque = sum(v["label"] for v in next(iter(oofs.values())).values()) if oofs else 0
    summary = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "labels": args.labels,
        "base_folds": os.path.relpath(BASE_FOLDS[args.labels], _ROOT),
        "majority_rate": n_masque / n if n else None,
        "n": n, "n_masque": n_masque,
        "full_baseline_acc": (sum(v["pred"] == v["label"] for v in full_oof.values()) / len(full_oof)
                              if full_oof else None),
        "conditions": results,
        "mcnemar_families": families,
        "note": "p_holm_within_family 只在本文件声明的族内校正; 论文表 I 的族需另算。",
    }
    if selrule_util.SELRULE:
        summary["selection_rule"] = selrule_util.RULE_DESC
        summary["full_baseline_dir"] = os.path.relpath(BASE_OUT[args.labels], _ROOT)
    os.makedirs(out_root(args.labels), exist_ok=True)
    path = os.path.join(out_root(args.labels), "summary.json")
    if args.only and not args.summary:
        path = os.path.join(out_root(args.labels), "summary_only_" + "_".join(c[0] for c in conds) + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    maj = summary["majority_rate"] * 100 if summary["majority_rate"] else float("nan")
    print(f"\n{'='*96}\nICOIN r2  labels={args.labels}  n={n}  majority={maj:.2f}%")
    for cid, s in results.items():
        folds_s = " ".join("%.1f" % (f["acc"] * 100) for f in s["per_fold"])
        print(f"  {cid:<20}{s['acc']*100:7.2f}%  [{s['ci95'][0]*100:.2f}, {s['ci95'][1]*100:.2f}]  "
              f"cm={s['cm']}  folds={folds_s}")
    for fname, fam in families.items():
        print(f"  -- family {fname}")
        for r in fam:
            print(f"     {r['A']:>18} vs {r['B']:<18} b={r['b_only_A_correct']:<4} c={r['c_only_B_correct']:<4} "
                  f"p={r['p']:.3g}  p_holm={r['p_holm_within_family']:.3g}")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
