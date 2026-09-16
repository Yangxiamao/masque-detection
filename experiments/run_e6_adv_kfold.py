# -*- coding: utf-8 -*-
"""[2026-09-08] E6-D: 把攻防实验从单一 403/50/51 切分迁到 5 折交叉验证。

## 为什么必须迁

论文 §V 的全部结论 —— padding 68.6 -> 94.1、pad+shape 43.1 -> 92.2、
"从未见过 shaped 样本的检测器迁移过来分数完全相同" —— 都压在【51 条测试流】上,
一条流 = 1.96 pp。于是:
  * "94.1 vs 92.2" 是一条流的差;
  * "两个检测器的混淆矩阵逐格相同"在 n=51 上并不是强证据;
  * 51 条里按比例含约 1 条标签噪声流, 占了误差预算的三分之一。

论文给的理由是"这一族需要每条会话的干净版与防御版配对, 所以单切分不可避免"。
**这个理由不成立**: 变换本来就在切分之后施加 (verify_obfuscation_bi.py 的断言),
那么在每一折内部先切分、再施加变换, 配对性完全保持。本脚本就是这么做的。

## 设计

折分配复用 outputs/e3_kfold/fold_assignment.json (与 E3 逐条相同), 于是:
  * 每条会话恰好被预测一次 -> 504 条 OOF, Wilson CI;
  * 与 E3 的 full 条件天然配对 -> McNemar 可用;
  * 干净版与防御版永远同折, 绝不跨折。

检测器 (每折各训一个, 从公开预训练 ckpt 重新微调):
  clean     训练集 = 干净 (直接复用 E3 的 full 条件, 见 --reuse-full)
  advpad    训练集 = 干净 + padding      (两类都加, 否则学到"有 padding = MASQUE")
  advshape  训练集 = 干净 + padding+shaping

评测变体 (对每折的 test 施加, 训练侧不动):
  T1 clean / T2 pad / T2-strict / T3 pad+shape / T3+absdir / T3+nointervals

    python experiments/run_e6_adv_kfold.py --stage data     # 只造数据
    python experiments/run_e6_adv_kfold.py --stage train    # 训 10 折
    python experiments/run_e6_adv_kfold.py --stage eval     # 90 次评测
    python experiments/run_e6_adv_kfold.py --stage summary
"""

import argparse
import copy
import json
import math
import os
import subprocess
import sys
import time

_ROOT = os.environ.get("MASQUE_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "pipeline", "pcap_split"))

from simulate_obfuscation_bi import TRANSFORMS  # noqa: E402

PROC = os.path.join(_ROOT, "data", "processed")
SRC = os.path.join(PROC, "quic_vs_masque_bi_feat")
FOLD_ROOT = os.path.join(PROC, "e6_adv_folds")
EVAL_ROOT = os.path.join(PROC, "e6_adv_evals")
OUT_ROOT = os.path.join(_ROOT, "outputs", "e6_adv_kfold")
ASSIGNMENT = os.path.join(_ROOT, "outputs", "e3_kfold", "fold_assignment.json")
E3_FULL = os.path.join(_ROOT, "outputs", "e3_kfold", "full")

N_SPLITS = 5

# 超参: 与 run_kfold_e3.py 逐字相同 (单变量仍只有 dataset transform)
HP = dict(num_packet=5, num_packet_byte=320, seq_len=20, stride_size=4,
          batch_size=16, epochs=30, model="fuse3_mamba_classifier",
          dataset_type="byte_size_interval", size_key="signed_sizes",
          interval_encoding="log10_decade")

# 检测器: id -> (训练集里混入的防御变换, 施加对象)
#   both        = 两类都施加 (正确做法)
#   masque_only = 只对 MASQUE 施加 —— 论文 §V-D 那个捷径的复现, 应当学到
#                 "有 padding = MASQUE" 而不是"是隧道"
DETECTORS = {
    "advpad":   (["pad"], "both"),
    "advshape": (["pad", "shape"], "both"),
    "advmo":    (["pad"], "masque_only"),
}

# 评测变体: id -> (对 test 施加的变换链, 施加对象)
VARIANTS = {
    "t1_clean":    ([], "both"),
    "t_shape":     (["shape"], "both"),
    "t2_pad":      (["pad"], "both"),
    "t2_strict":   (["pad", "nobytes"], "both"),
    "t3_padshape": (["pad", "shape"], "both"),
    "t3_absdir":   (["pad", "shape", "absdir"], "both"),
    "t3_noint":    (["pad", "shape", "nointervals"], "both"),
    # 只有 MASQUE 被 padding 的评测集: 捷径检测器在这里应当仍是满分,
    # 与它在 t2_pad 上的崩塌构成对照
    "t2_pad_mo":   (["pad"], "masque_only"),
}

MAJORITY = 270 / 504


# --------------------------------------------------------------------------- #
# 数据
# --------------------------------------------------------------------------- #

def flow_key(it):
    return os.path.basename(it["pcap_file"])


def load_src_items():
    items = []
    for sp in ["train", "valid", "test"]:
        with open(os.path.join(SRC, f"data-{sp}.json"), encoding="utf-8") as f:
            items += json.load(f)
    return {flow_key(it): it for it in items}


def apply_chain(item, chain, targets="both"):
    """targets='masque_only' 时只对 label==1 施加, 用于复现单类防御的捷径。"""
    if targets == "masque_only" and item["label"] != 1:
        return copy.deepcopy(item)
    out = item
    for name in chain:
        out = TRANSFORMS[name](out)
    return out


def split_fold(by_key, assign, k):
    """返回该折的 (train, valid, test) 干净条目, 与 split_data_kfold.py 逻辑一致。"""
    valid_keys = set(assign["valid_in_fold"][str(k)])
    tr, va, te = [], [], []
    for key, it in by_key.items():
        if assign["test_fold"][key] == k:
            te.append(it)
        elif key in valid_keys:
            va.append(it)
        else:
            tr.append(it)
    srt = lambda lst: sorted(lst, key=flow_key)  # noqa: E731
    return srt(tr), srt(va), srt(te)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def tag_variant(items, tag):
    out = []
    for it in items:
        c = copy.deepcopy(it)
        c["variant"] = tag
        out.append(c)
    return out


def build_data(assign, by_key, meta):
    """造 10 个训练折目录 + 30 个评测折目录。"""
    print(f"造数据: {len(by_key)} 条流, {N_SPLITS} 折")
    stats = {"train_dirs": {}, "eval_dirs": {}}

    for k in range(N_SPLITS):
        tr, va, te = split_fold(by_key, assign, k)

        # --- 训练折: 干净 + 防御版混合 (test 保持干净, 训练脚本不看它) ---
        for det, (chain, tgt) in DETECTORS.items():
            d = os.path.join(FOLD_ROOT, det, f"fold{k}")
            mix_tr = tag_variant(tr, "clean") + tag_variant(
                [apply_chain(it, chain, tgt) for it in tr], "+".join(chain))
            mix_va = tag_variant(va, "clean") + tag_variant(
                [apply_chain(it, chain, tgt) for it in va], "+".join(chain))
            write_json(os.path.join(d, "data-train.json"), mix_tr)
            write_json(os.path.join(d, "data-valid.json"), mix_va)
            write_json(os.path.join(d, "data-test.json"), tag_variant(te, "clean"))
            write_json(os.path.join(d, "metadata.json"), meta)
            stats["train_dirs"][f"{det}/fold{k}"] = {
                "train": len(mix_tr), "valid": len(mix_va), "test": len(te)}

        # --- 评测折: 只需 data-test.json + metadata.json ---
        for var, (chain, tgt) in VARIANTS.items():
            d = os.path.join(EVAL_ROOT, var, f"fold{k}")
            items = tag_variant([apply_chain(it, chain, tgt) for it in te],
                                "+".join(chain) or "clean")
            write_json(os.path.join(d, "data-test.json"), items)
            write_json(os.path.join(d, "metadata.json"), meta)
            stats["eval_dirs"][f"{var}/fold{k}"] = len(items)

    # ---------------- 断言 ----------------
    n_fail = 0

    def chk(cond, msg):
        nonlocal n_fail
        if not cond:
            n_fail += 1
            print(f"  [FAIL] {msg}")

    # 1. 每折的 train/valid/test 三集合 key 互不相交, 且并集 = 全集
    for k in range(N_SPLITS):
        tr, va, te = split_fold(by_key, assign, k)
        ktr, kva, kte = ({flow_key(x) for x in s} for s in (tr, va, te))
        chk(not (ktr & kva) and not (ktr & kte) and not (kva & kte),
            f"fold{k}: train/valid/test 有交集")
        chk(len(ktr | kva | kte) == len(by_key), f"fold{k}: 三集合并集 != 全集")

    # 2. 每条会话在 5 折里恰好当一次 test -> OOF 每条预测一次
    seen = {}
    for k in range(N_SPLITS):
        for it in split_fold(by_key, assign, k)[2]:
            seen[flow_key(it)] = seen.get(flow_key(it), 0) + 1
    chk(set(seen.values()) == {1}, "有会话被当作 test 的次数 != 1")
    chk(len(seen) == len(by_key), "OOF 覆盖的会话数 != 全集")

    # 3. 对抗训练集里 (label, variant) 的【条数】必须平衡 —— 否则"有防御"与
    #    "是 MASQUE"相关, 模型会学到检测 padding 而不是检测隧道 (论文 §V-D 的捷径)。
    #    注意 advmo 是【故意】违反这一点的对照组: 它的条数仍然平衡 (Direct 的
    #    "defended" 副本与 clean 副本逐字相同), 但特征层面只有 MASQUE 被 padding,
    #    所以捷径依然成立。这与已发表实现 (simulate_obfuscation_bi.py 的
    #    targets=masque_only + mix_clean) 的行为一致, 故本断言对它同样应当通过。
    for det in DETECTORS:
        for k in range(N_SPLITS):
            with open(os.path.join(FOLD_ROOT, det, f"fold{k}", "data-train.json"),
                      encoding="utf-8") as f:
                tr = json.load(f)
            cnt = {}
            for it in tr:
                cnt[(it["label"], it["variant"])] = cnt.get((it["label"], it["variant"]), 0) + 1
            per_label = {}
            for (lab, var), c in cnt.items():
                per_label.setdefault(lab, {})[var] = c
            for lab, d in per_label.items():
                vals = sorted(d.values())
                chk(len(set(vals)) == 1,
                    f"{det}/fold{k}: label={lab} 的 clean/defended 条数不等 {d}")

    # 4. 评测变体确实改了它该改的东西
    for k in range(N_SPLITS):
        with open(os.path.join(EVAL_ROOT, "t2_pad", f"fold{k}", "data-test.json"),
                  encoding="utf-8") as f:
            pad = json.load(f)
        chk(all(abs(int(s)) == 1350 for it in pad for s in it["signed_sizes"].split()),
            f"t2_pad/fold{k}: 存在 |size| != 1350")
        with open(os.path.join(EVAL_ROOT, "t3_absdir", f"fold{k}", "data-test.json"),
                  encoding="utf-8") as f:
            ad = json.load(f)
        chk(all(int(s) >= 0 for it in ad for s in it["signed_sizes"].split()),
            f"t3_absdir/fold{k}: 方向未抹净")

    print(f"  断言检查: {'全部通过' if n_fail == 0 else f'{n_fail} 条失败'}")
    if n_fail:
        sys.exit(1)
    write_json(os.path.join(OUT_ROOT, "data_stats.json"), stats)
    ex = stats["train_dirs"]["advpad/fold0"]
    print(f"  例: advpad/fold0 train={ex['train']} valid={ex['valid']} test={ex['test']}")


# --------------------------------------------------------------------------- #
# 训练 / 评测
# --------------------------------------------------------------------------- #

def train_one(det, k, epochs):
    data_dir = os.path.join(FOLD_ROOT, det, f"fold{k}")
    out_dir = os.path.join(OUT_ROOT, det, f"fold{k}")
    ckpt = os.path.join(_ROOT, "NetMambaPlus", "dataset", "checkpoints",
                        "pre-trained checkpoints", "fuse3_mamba.pth")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(data_dir, "metadata.json"), encoding="utf-8") as f:
        nb = len(json.load(f)["name_to_idx"])

    cmd = f"""
python -u "{os.path.normpath(os.path.join(_ROOT, "NetMambaPlus", "src", "fine-tune.py"))}" \\
    --num_packet {HP['num_packet']} --num_packet_byte {HP['num_packet_byte']} \\
    --seq_len {HP['seq_len']} --stride_size {HP['stride_size']} \\
    --batch_size {HP['batch_size']} --epochs {epochs} --seed 0 --nb_classes {nb} \\
    --finetune "{ckpt}" --data_path "{data_dir}" \\
    --output_dir "{out_dir}" --log_dir "{out_dir}" \\
    --model {HP['model']} --dataset_type {HP['dataset_type']} \\
    --size_key {HP['size_key']} --interval_encoding {HP['interval_encoding']} --no_amp
"""
    t0 = time.time()
    p = subprocess.run(cmd, shell=True,
                       cwd=os.path.join(_ROOT, "NetMambaPlus", "src", "scripts"))
    if p.returncode != 0:
        sys.exit(f"训练失败: {det} fold{k}")
    print(f"--- {det} fold{k} 用时 {time.time()-t0:.1f}s", flush=True)


def ckpt_path(det, k):
    if det == "clean":
        return os.path.join(E3_FULL, f"fold{k}", "checkpoint-best.pth")
    return os.path.join(OUT_ROOT, det, f"fold{k}", "checkpoint-best.pth")


def eval_one(det, var, k):
    out = os.path.join(OUT_ROOT, "preds", det, var, f"fold{k}.json")
    if os.path.exists(out):
        return True
    os.makedirs(os.path.dirname(out), exist_ok=True)
    p = subprocess.run(
        [sys.executable, os.path.join(_ROOT, "experiments", "eval_dump_preds.py"),
         "--data_dir", os.path.join(EVAL_ROOT, var, f"fold{k}"),
         "--ckpt", ckpt_path(det, k), "--out", out],
        capture_output=True, text=True)
    if p.returncode != 0:
        print(f"  !!! eval 失败 {det}/{var}/fold{k}: {p.stderr.strip()[-300:]}")
        return False
    return True


# --------------------------------------------------------------------------- #
# 汇总
# --------------------------------------------------------------------------- #

def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def load_oof(det, var):
    oof = {}
    for k in range(N_SPLITS):
        p = os.path.join(OUT_ROOT, "preds", det, var, f"fold{k}.json")
        if not os.path.exists(p):
            return None
        with open(p, encoding="utf-8") as f:
            oof.update(json.load(f))
    return oof


def summarize():
    from scipy.stats import binomtest

    dets = ["clean"] + list(DETECTORS)
    table, oofs = {}, {}
    for det in dets:
        for var in VARIANTS:
            oof = load_oof(det, var)
            if oof is None:
                continue
            n = len(oof)
            ok = sum(v["pred"] == v["label"] for v in oof.values())
            cm = [[0, 0], [0, 0]]
            for v in oof.values():
                cm[v["label"]][v["pred"]] += 1
            lo, hi = wilson(ok, n)
            table[f"{det}|{var}"] = {"detector": det, "variant": var, "n": n,
                                     "correct": ok, "acc": ok / n,
                                     "ci95": [lo, hi], "cm": cm}
            oofs[(det, var)] = oof

    # 配对检验: 同一变体上, 干净训练 vs 各对抗训练检测器
    pairs, pvals = [], []
    for var in VARIANTS:
        for det in DETECTORS:
            if ("clean", var) not in oofs or (det, var) not in oofs:
                continue
            a, b = oofs[("clean", var)], oofs[(det, var)]
            keys = sorted(set(a) & set(b))
            bb = sum(1 for k in keys if a[k]["pred"] == a[k]["label"] and b[k]["pred"] != b[k]["label"])
            cc = sum(1 for k in keys if a[k]["pred"] != a[k]["label"] and b[k]["pred"] == b[k]["label"])
            p = binomtest(bb, bb + cc, 0.5).pvalue if bb + cc else 1.0
            pairs.append({"variant": var, "A": "clean", "B": det,
                          "acc_A": table[f"clean|{var}"]["acc"],
                          "acc_B": table[f"{det}|{var}"]["acc"],
                          "b_only_A": bb, "c_only_B": cc, "n_discordant": bb + cc, "p": p})
            pvals.append(p)
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m, prev = len(pvals), 0.0
    for rank, i in enumerate(order):
        prev = max(prev, min(1.0, (m - rank) * pvals[i]))
        pairs[i]["p_holm"] = prev

    # advpad 与 advshape 在 T3 上是否真的等价 (论文"迁移完全"的主张)
    transfer = None
    if ("advpad", "t3_padshape") in oofs and ("advshape", "t3_padshape") in oofs:
        a, b = oofs[("advpad", "t3_padshape")], oofs[("advshape", "t3_padshape")]
        keys = sorted(set(a) & set(b))
        agree = sum(1 for k in keys if a[k]["pred"] == b[k]["pred"])
        bb = sum(1 for k in keys if a[k]["pred"] == a[k]["label"] and b[k]["pred"] != b[k]["label"])
        cc = sum(1 for k in keys if a[k]["pred"] != a[k]["label"] and b[k]["pred"] == b[k]["label"])
        transfer = {"n": len(keys), "identical_predictions": agree,
                    "agreement": agree / len(keys),
                    "b_only_advpad": bb, "c_only_advshape": cc,
                    "p": binomtest(bb, bb + cc, 0.5).pvalue if bb + cc else 1.0}

    summary = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
               "protocol": "pooled out-of-fold over 5 folds, n=504 sessions; "
                           "transforms applied inside each fold after assignment",
               "fold_assignment": ASSIGNMENT,
               "hyperparams": HP, "majority": MAJORITY,
               "table": table, "mcnemar_clean_vs_adv": pairs,
               "transfer_advpad_vs_advshape_on_t3": transfer}
    write_json(os.path.join(OUT_ROOT, "summary_e6_adv_kfold.json"), summary)

    # ---- 打印 ----
    W = 108
    print(f"\n{'='*W}")
    print("E6-D  攻防实验 (5 折 OOF, n=504 会话; 对比论文 Table II 的 51 条单切分)")
    print(f"{'='*W}")
    hdr = f"{'检测器':<26}" + "".join(f"{v:>13}" for v in VARIANTS)
    print(hdr)
    print("-" * W)
    names = {"clean": "干净训练", "advpad": "对抗训练 padding", "advshape": "对抗训练 pad+shape"}
    for det in dets:
        row = f"{names.get(det, det):<24}"
        for var in VARIANTS:
            e = table.get(f"{det}|{var}")
            row += f"{e['acc']*100:>12.2f}%" if e else f"{'-':>13}"
        print(row)
    print("-" * W)
    print(f"{'多数类基线':<24}" + "".join(f"{MAJORITY*100:>12.2f}%" for _ in VARIANTS))

    print(f"\n95% Wilson 区间 (n=504):")
    for key, e in table.items():
        print(f"  {key:<26} {e['acc']*100:6.2f}% [{e['ci95'][0]*100:.2f}, {e['ci95'][1]*100:.2f}]"
              f"  cm={e['cm']}")

    if pairs:
        print(f"\n配对 McNemar (同一变体上 干净训练 vs 对抗训练; Holm 校正, 族大小 {len(pairs)}):")
        for r in pairs:
            print(f"  {r['variant']:<13} clean {r['acc_A']*100:6.2f}% vs {r['B']:<9} "
                  f"{r['acc_B']*100:6.2f}%   b={r['b_only_A']:>3} c={r['c_only_B']:>3}  "
                  f"p={r['p']:.2e}  Holm={r['p_holm']:.2e}")

    if transfer:
        t = transfer
        print(f"\n迁移性 (T3 上 advpad vs advshape, 论文称'混淆矩阵逐格相同'):")
        print(f"  逐样本预测一致 {t['identical_predictions']}/{t['n']} = {t['agreement']*100:.2f}%"
              f"   b={t['b_only_advpad']} c={t['c_only_advshape']}  p={t['p']:.3f}")

    print(f"\n已写入 {os.path.join(OUT_ROOT, 'summary_e6_adv_kfold.json')}")


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "data", "train", "eval", "summary"])
    ap.add_argument("--epochs", type=int, default=HP["epochs"])
    args = ap.parse_args()

    with open(ASSIGNMENT, encoding="utf-8") as f:
        assign = json.load(f)
    with open(os.path.join(SRC, "metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    by_key = load_src_items()
    if set(by_key) != set(assign["test_fold"]):
        sys.exit("源数据集与折分配的样本集合不一致")

    if args.stage in ("all", "data"):
        build_data(assign, by_key, meta)

    if args.stage in ("all", "train"):
        todo = [(d, k) for d in DETECTORS for k in range(N_SPLITS)]
        for i, (det, k) in enumerate(todo, 1):
            if os.path.exists(ckpt_path(det, k)):
                print(f"[{i}/{len(todo)}] 跳过已训练: {det} fold{k}")
                continue
            print(f"[{i}/{len(todo)}] 训练 {det} fold{k}")
            train_one(det, k, args.epochs)

    if args.stage in ("all", "eval"):
        missing = [k for k in range(N_SPLITS) if not os.path.exists(ckpt_path("clean", k))]
        if missing:
            print(f"⚠️ E3 full 的 fold {missing} 没有 checkpoint (跑完就删了)。\n"
                  f"   先补: python experiments/run_kfold_e3.py --only full --keep-ckpt --force")
        todo = [(d, v, k) for d in (["clean"] + list(DETECTORS))
                for v in VARIANTS for k in range(N_SPLITS)
                if os.path.exists(ckpt_path(d, k))]
        ok = 0
        for i, (det, var, k) in enumerate(todo, 1):
            if eval_one(det, var, k):
                ok += 1
            if i % 15 == 0:
                print(f"  评测进度 {i}/{len(todo)}")
        print(f"评测完成 {ok}/{len(todo)}")

    if args.stage in ("all", "summary"):
        summarize()


if __name__ == "__main__":
    main()
