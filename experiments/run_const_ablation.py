"""[2026-08-18] E1: 常量消融 —— NetMamba+ 的 100% 有多少来自三个部署配置常量?

背景 (docs/01_Project_Management/NEXT_four_experiments_handoff.md §1.2 / §4):
报头几何分析显示 Direct/MASQUE 的可分性主要落在三个**与协议结构无关的 Cloudflare
部署配置常量**上 —— 最大包 1200 vs 1250、最小包 43 vs 24 (|DCID| 20 vs 0)、
是否发 Retry。单条零训练规则 "有 Retry" 在全 504 条流上已达 97.82%。
若 NetMamba+ 的 100% 也主要来自这三个常量，论文"结构性冗余"的主张就要改写。

数据集由 pipeline/pcap_split/ablate_constants.py 生成，变换 label-free 且保号，
施加于 train/valid/test 全部 split，每组**从头单独重训**。

条件
----
c1            MTU 归一化 (每流最大包 -> 1292)
c2            CID 归一化 (按方向把最小包平移到 63)
c3            去掉真实 Retry 帧 (严谨版, 帧序号来自 QUIC 长包头 ptype==3)
c123          三者同时  ★主指标
c123_nobytes  三者同时 + 遮蔽字节通道 (诊断: 常量是否从 data 的原始报头泄回来
              —— 首包长包头里 DCID 长度和 IP total length 是明文可读的)

超参与 §3.4 / run_iat_encoding_fix.py 严格一致，单变量只有数据集。
⚠️ --interval_encoding log10_decade 必须显式传, 默认值仍是有缺陷的 sigmoid_log。

    python experiments/run_const_ablation.py            # 训练 5 组 + 汇总
    python experiments/run_const_ablation.py --summary  # 只汇总已有结果
"""

import argparse
import json
import os
import subprocess
import sys
import time

_ROOT = os.environ.get("MASQUE_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BYTE_PACKET_NUM = 5
SEQ_PACKET_NUM = 20
HEADER_BYTE_NUM = 80
PAYLOAD_BYTE_NUM = 240
STRIDE_LEN = 4
INTERVAL_ENCODING = "log10_decade"

# (条件 id, 数据集目录, 输出目录, 汇报用名称)
CONDITIONS = [
    ("c1",           "bi_const_c1",           "const_c1",
     "C1 MTU 归一化 (最大包 -> 1292)"),
    ("c2",           "bi_const_c2",           "const_c2",
     "C2 CID 归一化 (最小包 -> 63)"),
    ("c3",           "bi_const_c3",           "const_c3",
     "C3 去 Retry 帧 (严谨版)"),
    ("c123",         "bi_const_c123",         "const_c123",
     "C123 三者同时 ★主指标"),
    ("c123_nobytes", "bi_const_c123_nobytes", "const_c123_nobytes",
     "C123 + No-Bytes (诊断字节通道)"),
]

# 参照基线 (2026-08-17 U2, 同超参同切分)
BASELINES = [
    ("Full 全特征基线", "quic_vs_masque_bi_feat", "iat_enc_full_new"),
    ("No-Sizes+No-Bytes (只剩时间)", "bi_abl_no_sizes_nobytes", "iat_enc_nsnb_new"),
]

# 零训练单规则 (描述性, 全 504 条流, 见 outputs/header_geometry/)
RULE_REFERENCE = {"max_eq_1200": 0.9583, "min_ge_42": 0.9504, "has_retry": 0.9782}


def train(data_name, out_name, tag, epochs=30, cuda_idx=0):
    data_dir = os.path.join(_ROOT, "data", "processed", data_name)
    output_dir = os.path.join(_ROOT, "outputs", out_name)
    ckpt_path = os.path.join(_ROOT, "NetMambaPlus", "dataset", "checkpoints",
                             "pre-trained checkpoints", "fuse3_mamba.pth")
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(data_dir, "metadata.json"), encoding="utf-8") as f:
        num_class = len(json.load(f)["name_to_idx"])

    print(f"\n{'='*74}\n训练条件: {tag}\n  data     = {data_dir}\n"
          f"  encoding = {INTERVAL_ENCODING}\n  output   = {output_dir}\n{'='*74}", flush=True)

    cmd = f"""
python -u "{os.path.normpath(os.path.join(_ROOT, "NetMambaPlus", "src", "fine-tune.py"))}" \\
    --num_packet {BYTE_PACKET_NUM} \\
    --num_packet_byte {HEADER_BYTE_NUM + PAYLOAD_BYTE_NUM} \\
    --seq_len {SEQ_PACKET_NUM} \\
    --stride_size {STRIDE_LEN} \\
    --batch_size 16 \\
    --epochs {epochs} \\
    --nb_classes {num_class} \\
    --finetune "{ckpt_path}" \\
    --data_path "{data_dir}" \\
    --output_dir "{output_dir}" \\
    --log_dir "{output_dir}" \\
    --model fuse3_mamba_classifier \\
    --dataset_type byte_size_interval \\
    --size_key signed_sizes \\
    --interval_encoding {INTERVAL_ENCODING} \\
    --no_amp
"""
    env = os.environ.copy()
    if sys.platform != "win32":
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_idx)
    t0 = time.time()
    proc = subprocess.run(cmd, shell=True, env=env,
                          cwd=os.path.join(_ROOT, "NetMambaPlus", "src", "scripts"))
    if proc.returncode != 0:
        print(f"!!! 训练失败: {tag}")
        sys.exit(proc.returncode)
    print(f"--- {tag} 用时 {time.time()-t0:.1f}s", flush=True)


def read_stats(out_name):
    p = os.path.join(_ROOT, "outputs", out_name, "test_stats.json")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true", help="跳过训练, 只汇总")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--only", type=str, default=None, help="只跑指定条件 id (逗号分隔)")
    args = ap.parse_args()

    conds = CONDITIONS
    if args.only:
        want = {s.strip() for s in args.only.split(",")}
        conds = [c for c in CONDITIONS if c[0] in want]

    if not args.summary:
        for cid, data_name, out_name, tag in conds:
            train(data_name, out_name, tag, epochs=args.epochs)

    base_rows, rows = [], []
    for tag, data_name, out_name in BASELINES:
        try:
            s = read_stats(out_name)
        except FileNotFoundError:
            continue
        base_rows.append({"tag": tag, "data": data_name, "acc": s["acc"],
                          "weighted_f1": s["weighted_f1"], "cm": s["cm"]})
    for cid, data_name, out_name, tag in CONDITIONS:
        try:
            s = read_stats(out_name)
        except FileNotFoundError:
            print(f"(跳过未训练的条件: {cid})")
            continue
        rows.append({"id": cid, "tag": tag, "data": data_name, "acc": s["acc"],
                     "weighted_f1": s["weighted_f1"], "cm": s["cm"]})

    out = os.path.join(_ROOT, "outputs", "summary_const_ablation.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"baselines": base_rows, "conditions": rows,
                   "unfitted_rule_reference_n504": RULE_REFERENCE}, f,
                  indent=2, ensure_ascii=False)

    print(f"\n\n{'='*92}")
    print("E1 常量消融 (双向 + 会话级切分 + signed_sizes + log10_decade, n=51 test sessions)")
    print(f"{'='*92}")
    print(f"{'条件':<40}{'Accuracy':>11}{'F1':>10}   混淆矩阵")
    print("-" * 92)
    for r in base_rows:
        print(f"{'[基线] ' + r['tag']:<38}{r['acc']*100:>10.2f}%{r['weighted_f1']:>10.4f}   {r['cm']}")
    print("-" * 92)
    for r in rows:
        print(f"{r['tag']:<38}{r['acc']*100:>10.2f}%{r['weighted_f1']:>10.4f}   {r['cm']}")
    print("-" * 92)
    full = next((b["acc"] for b in base_rows if b["tag"].startswith("Full")), None)
    main_acc = next((r["acc"] for r in rows if r["id"] == "c123"), None)
    if full is not None and main_acc is not None:
        print(f"主指标 C123 vs Full 基线: {full*100:.2f}% -> {main_acc*100:.2f}% "
              f"({(main_acc-full)*100:+.2f} pp)")
    print("\n参照 (零训练单规则, 全 504 条流, 描述性未交叉验证):")
    for k, v in RULE_REFERENCE.items():
        print(f"  {k:<16}{v*100:.2f}%")
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
