"""[2026-10-12 ICOIN r2 #3] 改标版折数据集: 11 条流 label -> 0 / name -> Direct, 其余逐字不变。

见 docs/01_Planning/Handoff_ICOIN_Revision_2026-10-12.md §3 #3。
这 11 条 (outputs/e3_kfold/error_analysis_e3.json 的 label_noise_flows) 的对端不在
WARP 162.159 前缀, 且握手形态 (首包 1292 B、客户端 DCID 长 8、无 Retry) 是 Chromium
直连的特征 (outputs/v3_kfold/single_feature_diag.json v1.masque_with_first_size_1292)。

做法: 递归复制 <src_root> 下的全部 data-*.json / metadata.json 到 <dst_root>,
只改这 11 条的 label 与 name。样本集合、折分配、变换结果都与原版逐字相同,
唯一变量是标签 —— 原标签与改标结果可做配对 McNemar。

    python pipeline/pcap_split/make_relabel_folds.py \
        --src_root data/processed/e3_folds --dst_root data/processed/e3r_folds [--only full,no_bytes]

⚠️ 只写 dst_root, 绝不改 src_root。dst 已存在的文件默认跳过 (--force 覆盖)。
"""

import argparse
import json
import os
import shutil

_ROOT = os.environ.get("MASQUE_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ERR = os.path.join(_ROOT, "outputs", "e3_kfold", "error_analysis_e3.json")
N_NOISE = 11


def load_noise():
    with open(ERR, encoding="utf-8") as f:
        noise = set(json.load(f)["label_noise_flows"])
    assert len(noise) == N_NOISE, f"噪声流应为 {N_NOISE} 条, 实为 {len(noise)}"
    return noise


def relabel_file(src, dst, noise, direct_idx, direct_name):
    with open(src, encoding="utf-8") as f:
        items = json.load(f)
    n_changed = 0
    for it in items:
        if os.path.basename(it["pcap_file"]) in noise:
            assert it["label"] == 1, f"噪声流原标签应为 1: {it['pcap_file']}"
            it["label"] = direct_idx
            it["name"] = direct_name
            n_changed += 1
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)
    return n_changed, len(items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root", required=True)
    ap.add_argument("--dst_root", required=True)
    ap.add_argument("--only", default=None, help="只处理 src_root 下这些一级子目录 (逗号分隔)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    src_root = args.src_root if os.path.isabs(args.src_root) else os.path.join(_ROOT, args.src_root)
    dst_root = args.dst_root if os.path.isabs(args.dst_root) else os.path.join(_ROOT, args.dst_root)
    assert os.path.abspath(src_root) != os.path.abspath(dst_root), "dst 不能等于 src"
    noise = load_noise()
    subdirs = ([s.strip() for s in args.only.split(",")] if args.only
               else sorted(d for d in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, d))))

    for sub in subdirs:
        tot_changed = 0
        per_dir = {}
        for dirpath, _, files in os.walk(os.path.join(src_root, sub)):
            rel = os.path.relpath(dirpath, src_root)
            meta_path = os.path.join(dirpath, "metadata.json")
            if not os.path.exists(meta_path):
                continue
            with open(meta_path, encoding="utf-8") as f:
                n2i = json.load(f)["name_to_idx"]
            direct_name = next(n for n, i in n2i.items() if i == 0)
            assert direct_name.lower().startswith("direct"), n2i
            for fn in sorted(files):
                src = os.path.join(dirpath, fn)
                dst = os.path.join(dst_root, rel, fn)
                if os.path.exists(dst) and not args.force:
                    continue
                if fn == "metadata.json":
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy(src, dst)
                elif fn.startswith("data-") and fn.endswith(".json"):
                    c, n = relabel_file(src, dst, noise, 0, direct_name)
                    tot_changed += c
                    per_dir[rel] = per_dir.get(rel, 0) + c
        # 每个含 test 的折目录里, 11 条流在 train/valid/test 合计恰好出现 (变换后的条数 / 504) 倍
        print(f"{sub:<24} 改标 {tot_changed} 条记录, {len(per_dir)} 个目录")


if __name__ == "__main__":
    main()
