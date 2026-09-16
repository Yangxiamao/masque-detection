"""[2026-08-23] E4: 按对端分组构造「留出组」数据集, 检验模型是否只在认「对端是不是 Cloudflare」。

回答什么 (NEXT_four_experiments_handoff.md §7):
  E4-A  训练集里完全没有 Direct->Cloudflare 的流, 测试集全是它们。
        主指标 = 这批流仍被判为 Direct 的比例。
  E4-B  按 Direct 侧对端 /16 做 leave-one-group-out (非对称设计: 只对 Direct 侧分组,
        MASQUE 侧照常随机 —— 因为 MASQUE 的对端 96% 都是 162.159, 对称分组不可行,
        见 E3_kfold_design.md §1.3)。

⚠️ 前提: 特征提取的去偏是 [relative_zero_ip, zero_port], 真实 IP 已被映射成
   0.0.0.0 / 0.0.0.1 (只保留方向)。**模型读不到对端 IP 本身。**
   所以 E4 检验的不是「模型是否直接读 IP」, 而是「模型是否学到了与对端强相关的
   站点级流量指纹, 而非隧道指纹」。

三个模式:
  cf_holdout    test 的 Direct = 全部 49 条对端在 Cloudflare 段的流 (train 里一条都没有)
  nocf_holdout  对照组: test 的 Direct = 等量 (49 条) 随机抽的非 Cloudflare Direct,
                且 49 条 CF Direct 全部留在 train。用来分离「Cloudflare」与「换一批流」
  groupout:<g>  留出 Direct 侧某个 /16 组 (E4-B)

三个模式的 MASQUE 侧一律: 从 259 条 WARP 入口 (162.159/16) 流里随机抽等量入 test,
其余 (含 11 条已知标签噪声流) 入 train。抽样种子固定, 逐模式记录。

    python pipeline/split_data_e4.py --mode cf_holdout
    python pipeline/split_data_e4.py --mode nocf_holdout
    python pipeline/split_data_e4.py --mode groupout:104.18
    python pipeline/split_data_e4.py --all          # 一次造齐 2 + 7 个
"""

import argparse
import collections
import json
import os
import random
import sys

_ROOT = os.environ.get("MASQUE_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "experiments"))

from analyze_peer_groups import load_flows  # 复用对端解析, 不复制实现

FEAT_DIR = os.path.join(_ROOT, "data", "processed", "quic_vs_masque_bi_feat")
OUT_ROOT = os.path.join(_ROOT, "data", "processed", "e4_groups")
SEED = 0
VALID_RATIO = 0.1
MIN_GROUP = 8  # E4-B 只对 >= 8 条流的 Direct /16 组做 leave-one-group-out


def load_feat_items(feat_dir=FEAT_DIR):
    items = []
    for sp in ("train", "valid", "test"):
        with open(os.path.join(feat_dir, f"data-{sp}.json"), encoding="utf-8") as f:
            items += json.load(f)
    by_key = {os.path.basename(it["pcap_file"]): it for it in items}
    if len(by_key) != len(items):
        raise SystemExit("basename 不唯一")
    return by_key


def pick_test_keys(rows, mode, rng):
    """返回 (test_direct_keys, test_masque_keys, 描述)。"""
    direct = [r for r in rows if r["name"] == "Direct"]
    masque_clean = [r for r in rows if r["name"] == "Masque" and r["is_warp_entry"]]

    if mode == "cf_holdout":
        td = [r["key"] for r in direct if r["is_cf"]]
        desc = "test Direct = 全部对端在 Cloudflare 段的 Direct 流"
    elif mode == "nocf_holdout":
        pool = [r["key"] for r in direct if not r["is_cf"]]
        n = sum(1 for r in direct if r["is_cf"])
        td = sorted(rng.sample(pool, n))
        desc = f"对照: test Direct = 随机 {n} 条非 Cloudflare Direct (CF 流全部留在 train)"
    elif mode.startswith("groupout:"):
        g = mode.split(":", 1)[1]
        td = [r["key"] for r in direct if r["peer16"] == g]
        if not td:
            raise SystemExit(f"没有对端 /16 = {g} 的 Direct 流")
        desc = f"留出 Direct 对端 /16 = {g} ({len(td)} 条)"
    else:
        raise SystemExit(f"未知模式: {mode}")

    tm = sorted(rng.sample([r["key"] for r in masque_clean], min(len(td), len(masque_clean))))
    return sorted(td), tm, desc


def build(mode, rows, by_key, out_root=OUT_ROOT):
    name = mode.replace(":", "_").replace(".", "_")
    rng = random.Random(SEED + sum(ord(c) for c in mode))  # 逐模式确定性种子

    td, tm, desc = pick_test_keys(rows, mode, rng)
    test_keys = set(td) | set(tm)
    rest = [k for k in sorted(by_key) if k not in test_keys]

    # valid: 从 rest 里按 label 分层抽 VALID_RATIO, 种子固定 (fine-tune.py 用 valid 选 best ckpt)
    by_label = collections.defaultdict(list)
    for k in rest:
        by_label[by_key[k]["label"]].append(k)
    valid_keys = set()
    for lab, ks in sorted(by_label.items()):
        n = max(1, int(round(len(ks) * VALID_RATIO)))
        valid_keys |= set(rng.sample(ks, n))
    train_keys = [k for k in rest if k not in valid_keys]

    info = {r["key"]: r for r in rows}

    # ---- 自检 (CLAUDE.md 规则 5; 造集合错了后面全废) ----
    assert not (set(train_keys) & valid_keys), "train/valid 相交"
    assert not (set(train_keys) & test_keys), "train/test 相交"
    assert not (valid_keys & test_keys), "valid/test 相交"
    assert len(train_keys) + len(valid_keys) + len(test_keys) == len(by_key), "三集合之和 != 504"
    assert valid_keys, "valid 为空"
    if mode == "cf_holdout":
        leak = [k for k in train_keys + sorted(valid_keys)
                if info[k]["name"] == "Direct" and info[k]["is_cf"]]
        assert not leak, f"cf_holdout 泄漏: train/valid 里还有 {len(leak)} 条 CF Direct"
    if mode.startswith("groupout:"):
        g = mode.split(":", 1)[1]
        leak = [k for k in train_keys + sorted(valid_keys)
                if info[k]["name"] == "Direct" and info[k]["peer16"] == g]
        assert not leak, f"groupout 泄漏: train/valid 里还有 {len(leak)} 条 {g} 的 Direct"

    d = os.path.join(out_root, name)
    os.makedirs(d, exist_ok=True)
    parts = {"train": sorted(train_keys), "valid": sorted(valid_keys),
             "test": sorted(test_keys)}
    for sp, ks in parts.items():
        with open(os.path.join(d, f"data-{sp}.json"), "w", encoding="utf-8") as f:
            json.dump([by_key[k] for k in ks], f, indent=2)
    with open(os.path.join(FEAT_DIR, "metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    with open(os.path.join(d, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    def cnt(ks):
        c = collections.Counter(info[k]["name"] for k in ks)
        return f"Direct {c['Direct']:3d} / Masque {c['Masque']:3d}"

    manifest = {
        "mode": mode, "desc": desc, "seed": SEED, "valid_ratio": VALID_RATIO,
        "n_train": len(train_keys), "n_valid": len(valid_keys), "n_test": len(test_keys),
        "test_direct_keys": td, "test_masque_keys": tm,
        "test_direct_by16": dict(collections.Counter(info[k]["peer16"] for k in td)),
        "train_direct_by16": dict(collections.Counter(
            info[k]["peer16"] for k in train_keys if info[k]["name"] == "Direct")),
        "data_dir": os.path.relpath(d, _ROOT).replace("\\", "/"),
    }
    with open(os.path.join(d, "e4_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[{mode}] {desc}")
    print(f"    train {len(train_keys):3d} ({cnt(train_keys)})   "
          f"valid {len(valid_keys):2d} ({cnt(sorted(valid_keys))})   "
          f"test {len(test_keys):3d} ({cnt(sorted(test_keys))})")
    print(f"    test 的 Direct 对端 /16: "
          f"{sorted(manifest['test_direct_by16'].items(), key=lambda x: -x[1])}")
    print(f"    -> {d}")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default=None,
                    help="cf_holdout | nocf_holdout | groupout:<peer /16>")
    ap.add_argument("--all", action="store_true",
                    help="造齐 E4-A 两组 + E4-B 全部 >=%d 条的 Direct /16 组" % MIN_GROUP)
    ap.add_argument("--out_root", default=OUT_ROOT)
    args = ap.parse_args()

    rows = load_flows()
    by_key = load_feat_items()
    assert len(rows) == len(by_key) == 504, f"流数不符: {len(rows)} / {len(by_key)}"

    if args.all:
        d16 = collections.Counter(r["peer16"] for r in rows if r["name"] == "Direct")
        groups = [g for g, k in d16.most_common() if k >= MIN_GROUP]
        modes = ["cf_holdout", "nocf_holdout"] + [f"groupout:{g}" for g in groups]
    elif args.mode:
        modes = [args.mode]
    else:
        raise SystemExit("给 --mode 或 --all")

    mans = [build(m, rows, by_key, args.out_root) for m in modes]

    p = os.path.join(_ROOT, "outputs", "e4_cloudflare", "e4_datasets.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(mans, f, indent=2, ensure_ascii=False)
    print(f"\n共 {len(mans)} 个数据集, 清单 -> {p}")


if __name__ == "__main__":
    main()
