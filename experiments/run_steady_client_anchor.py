# -*- coding: utf-8 -*-
"""[2026-10-16 ICOIN r4 #1] 稳态段 (第 51-100 包) 按真实 client 地址重定方向后重跑。

审稿意见 r4 #1: III-B 定义 "正 = client->server", 但 VI-A 稳态段的符号锚定在第 51 包的发送方
(448 条里 246 条是 server)。两组评估的方向定义不一致。修法见 pipeline/feat_extract/client_anchor.py。

受影响的 (用了带符号包长 / 方向哨兵 IP):
  steady_cli           稳态全特征                (旧: steady, 99.33%)
  steady_cli_suff_dir  仅方向 pad+nobytes+nointervals (旧: steady_suff_dir, 93.08%)
不受影响的 (输入与旧数据逐字节相同, stage_verify 核对后复用旧预测, 不重训):
  steady_suff_len  absdir+nobytes+nointervals
  steady_suff_time nosizes+nobytes
  head448          锚点本来就是 client (448/448)

折 = 旧的 outputs/e6_steady/fold_assignment_448.json, 文件名不变 -> 与旧条件逐条配对。
旧脚本 run_e6_steady.py / run_steady_suff.py 不动, 旧结果可复现。

WSL MAMBA 环境:
    python experiments/run_steady_client_anchor.py --stage all
输出 outputs/e6_steady_sel/summary_steady_client_anchor.json
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

os.environ["MASQUE_SELRULE"] = "acc_loss"

_ROOT = os.environ.get("MASQUE_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "experiments"))
sys.path.insert(0, os.path.join(_ROOT, "pipeline", "pcap_split"))

import run_kfold_e3 as _e3  # noqa: E402
import selrule_util  # noqa: E402
from run_kfold_e3 import load_oof, summarize, train_fold  # noqa: E402
from simulate_obfuscation_bi import generate  # noqa: E402

PROC = os.path.join(_ROOT, "data", "processed")
FOLD_ROOT = os.path.join(PROC, "e3_folds")
N_SPLITS = 5
OUT_ROOT = selrule_util.out_dir(os.path.join(_ROOT, "outputs", "e3_kfold"))
_e3.OUT_ROOT = OUT_ROOT
ASSIGNMENT = os.path.join(_ROOT, "outputs", "e6_steady", "fold_assignment_448.json")
SUMMARY = os.path.join(_ROOT, "outputs", "e6_steady_sel", "summary_steady_client_anchor.json")

PCAP_DIR = "quic_vs_masque_bi_seg50_50"
FEAT_DIR = "e6_steady_cli_feat"
FULL, FULL_OLD = "steady_cli", "steady"
DIR_TF = ["pad", "nobytes", "nointervals"]
# (新条件 id, 变换, 旧条件 id, 说明);  len / time 的新 id 为 None = 复用旧预测
SUFF = [
    ("steady_suff_len",     ["absdir", "nobytes", "nointervals"], "steady_suff_len",  "length only"),
    ("steady_cli_suff_dir", DIR_TF,                               "steady_suff_dir",  "direction only"),
    ("steady_suff_time",    ["nosizes", "nobytes"],               "steady_suff_time", "timing only"),
]
TRAIN = [FULL, "steady_cli_suff_dir"]


def sh(cmd):
    if subprocess.run(cmd).returncode != 0:
        sys.exit(f"命令失败: {cmd}")


def read_split(d, split):
    with open(os.path.join(d, f"data-{split}.json"), encoding="utf-8") as f:
        return json.load(f)


def by_name(d):
    out = {}
    for sp in ("train", "valid", "test"):
        for it in read_split(d, sp):
            out[os.path.basename(it["pcap_file"])] = (sp, it)
    return out


def stage_data():
    feat = os.path.join(PROC, FEAT_DIR)
    if not os.path.exists(os.path.join(feat, "data.json")):
        sh([sys.executable, os.path.join(_ROOT, "pipeline", "feat_extract", "feat_quic_vs_masque_bi.py"),
            "--pcap_dir", PCAP_DIR, "--save_dir", FEAT_DIR, "--anchor", "client"])
    if not os.path.exists(os.path.join(feat, "data-test.json")):
        sh([sys.executable, os.path.join(_ROOT, "pipeline", "split_data.py"), "--feat_dir", feat])
    assert os.path.exists(ASSIGNMENT), ASSIGNMENT
    sh([sys.executable, os.path.join(_ROOT, "pipeline", "split_data_kfold.py"),
        "--feat_dir", feat, "--out_root", os.path.join(FOLD_ROOT, FULL), "--assignment", ASSIGNMENT])
    for k in range(N_SPLITS):
        generate(os.path.join(FOLD_ROOT, FULL, f"fold{k}"),
                 os.path.join(FOLD_ROOT, "steady_cli_suff_dir", f"fold{k}"),
                 DIR_TF, "both", ["train", "valid", "test"], False)


def stage_verify():
    """核对: (1) 每折样本归属与旧折逐条相同; (2) 锚点为 client 的旧样本逐字节不变,
    锚点为 server 的旧样本 signed_sizes 恰好取反、intervals 不变; (3) 新折做 len / time
    变换后与旧 steady_suff_len / steady_suff_time 逐字节相同 -> 可复用旧预测。"""
    res = {"flipped": {"Direct": 0, "Masque": 0}, "kept": {"Direct": 0, "Masque": 0}}
    for k in range(N_SPLITS):
        new = by_name(os.path.join(FOLD_ROOT, FULL, f"fold{k}"))
        old = by_name(os.path.join(FOLD_ROOT, FULL_OLD, f"fold{k}"))
        assert set(new) == set(old), f"fold{k} 样本集合不同"
        for name, (sp, it) in new.items():
            osp, o = old[name]
            assert sp == osp, f"fold{k} {name}: split {sp} vs {osp}"
            assert it["label"] == o["label"] and it["intervals"] == o["intervals"], name
            ns = [int(x) for x in it["signed_sizes"].split()]
            os_ = [int(x) for x in o["signed_sizes"].split()]
            cls = it["name"]
            n_count = 1 if sp == "test" else 0     # 每条样本恰在一折的 test 里, 只计一次
            if ns[0] > 0:      # 第 51 包是 client 发的 -> 与旧管道相同
                assert it == o, f"{name}: 锚点相同但特征不同"
                res["kept"][cls] += n_count
            else:
                assert ns == [-x for x in os_], f"{name}: 符号不是整体取反"
                assert it["data"] != o["data"], f"{name}: 方向哨兵未改变"
                res["flipped"][cls] += n_count
    print(f"锚点核对: 旧锚点为 server (已翻正) {res['flipped']}, 本就是 client {res['kept']}")

    tmp = tempfile.mkdtemp(prefix="steady_cli_verify_")
    try:
        for cid, tf, old_cid, _ in SUFF:
            if cid != old_cid:
                continue
            for k in range(N_SPLITS):
                dst = os.path.join(tmp, cid, f"fold{k}")
                generate(os.path.join(FOLD_ROOT, FULL, f"fold{k}"), dst, tf, "both",
                         ["train", "valid", "test"], False, verbose=False)
                for sp in ("train", "valid", "test"):
                    assert read_split(dst, sp) == read_split(os.path.join(FOLD_ROOT, old_cid, f"fold{k}"), sp), \
                        f"{cid} fold{k} {sp}: 变换后与旧数据不同, 不能复用旧预测"
            print(f"  {cid}: 5 折变换后与旧数据逐字节相同 -> 复用旧预测")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return res


def stage_train(epochs):
    todo = [(cid, k) for cid in TRAIN for k in range(N_SPLITS)]
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


def mcnemar(ref, oof):
    """b = 仅 ref 对, c = 仅 oof 对; 精确二项检验。"""
    from scipy.stats import binomtest
    keys = sorted(set(ref) & set(oof))
    assert len(keys) == len(ref) == len(oof), "配对样本不全"
    b = sum(1 for x in keys if ref[x]["pred"] == ref[x]["label"] and oof[x]["pred"] != oof[x]["label"])
    c = sum(1 for x in keys if ref[x]["pred"] != ref[x]["label"] and oof[x]["pred"] == oof[x]["label"])
    return {"n": len(keys), "b_ref_only": b, "c_cond_only": c,
            "p": binomtest(b, b + c, 0.5).pvalue if b + c else 1.0}


def fmt(s):
    return f"{s['acc']*100:6.2f}% [{s['ci95'][0]*100:.2f}, {s['ci95'][1]*100:.2f}] cm {s['cm']}"


def stage_summary(verify):
    sys.path.insert(0, os.path.join(_ROOT, "experiments"))
    from run_steady_suff import rf_cond

    out = {"selection_rule": selrule_util.RULE_DESC, "anchor_check": verify, "conditions": {}}
    full, _ = load_oof(FULL, 0)
    full_old, _ = load_oof(FULL_OLD, 0)
    head, _ = load_oof("head448", 0)
    if full is None:
        sys.exit("steady_cli 未训练完")
    s = summarize(full)
    s["mcnemar_vs_head448"] = mcnemar(head, full)      # b = 仅头部对, c = 仅稳态对
    s["mcnemar_vs_old_anchor"] = mcnemar(full_old, full)
    s["old_anchor"] = summarize(full_old)
    out["conditions"][FULL] = s
    print(f"\n{'='*100}\n稳态段 (client 锚定)  n={s['n']}\n{'='*100}")
    print(f"full            NetMamba+ {fmt(s)}   (旧锚点 {fmt(s['old_anchor'])})")
    print(f"  vs head448: 仅头部对 {s['mcnemar_vs_head448']['b_ref_only']}, "
          f"仅稳态对 {s['mcnemar_vs_head448']['c_cond_only']}, p = {s['mcnemar_vs_head448']['p']:.4g}")

    for cid, tf, old_cid, note in SUFF:
        oof, per_fold = load_oof(cid, 0)
        if oof is None:
            print(f"(未完成: {cid})")
            continue
        c = summarize(oof)
        c["note"], c["transforms"], c["fold_accs"] = note, tf, [f["acc"] for f in per_fold]
        c["reused_old_preds"] = cid == old_cid
        c["mcnemar_vs_full"] = mcnemar(full, oof)
        c["rf"] = rf_cond(cid)
        if cid != old_cid:
            old, _ = load_oof(old_cid, 0)
            c["old_anchor"] = summarize(old)
            c["old_anchor_rf"] = rf_cond(old_cid)
        out["conditions"][cid] = c
        m = c["mcnemar_vs_full"]
        print(f"{note:<15} NetMamba+ {fmt(c)}  vs full b/c {m['b_ref_only']}/{m['c_cond_only']} "
              f"p {m['p']:.3g}   RF {c['rf']['acc']*100:6.2f}%"
              + (f"   (旧锚点 {c['old_anchor']['acc']*100:.2f}%, RF {c['old_anchor_rf']['acc']*100:.2f}%)"
                 if "old_anchor" in c else "   (复用旧预测)"))
    labels = [v["label"] for v in full.values()]
    out["majority_baseline"] = max(labels.count(0), labels.count(1)) / len(labels)
    print(f"多数类基线 {out['majority_baseline']*100:.2f}%")
    os.makedirs(os.path.dirname(SUMMARY), exist_ok=True)
    with open(SUMMARY, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"已写入 {SUMMARY}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "data", "verify", "train", "summary"])
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    if args.stage in ("all", "data"):
        stage_data()
    verify = None
    if args.stage in ("all", "verify", "summary"):
        verify = stage_verify()
    if args.stage in ("all", "train"):
        stage_train(args.epochs)
    if args.stage in ("all", "summary"):
        stage_summary(verify)


if __name__ == "__main__":
    main()
