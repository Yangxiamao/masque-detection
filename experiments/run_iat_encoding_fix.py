"""
[2026-08-17] U2: 修 IAT 编码后重训 NetMamba+ —— 检验 52.94% 是否为编码 artifact。

背景 (PROJECT_LOG 2026-07-20 第三条):
  loader_data.py 原先对 IAT 施加 sigmoid(log(1+x)) = (1+x)/(2+x), 把 [0, inf) 压进
  [0.5, 1.0]。该函数在亚毫秒区几乎零分辨率, 而本数据集的判别信号在百微秒量级
  → 编码后类间差仅占值域的 0.0014%, 下游 FixedCosineEmbed (为 0~10000 整数位置
  设计的正弦编码) 收到的 20 个 token embedding 平均余弦相似度 0.999997。
  同期 IAT 统计基线证明: 同样的前 20 包 + 同样的会话级切分下,
  随机森林可达 100% (test) / 96.83% +- 2.76% (5-fold, n=504)。

本实验把编码换成 log10_decade (100 units/decade, 见 loader_data.encode_intervals),
在**完全相同的超参与会话级切分**下重训, 单变量只有 --interval_encoding。

四个条件:
  1. nsnb_old      No-Sizes+No-Bytes, 旧编码  -> 复现对照, 期望 52.94%
  2. nsnb_new      No-Sizes+No-Bytes, 新编码  -> 主命题: 只剩时间时能否接近 RF 的 ~97%
  3. full_new      全特征基线,        新编码  -> 回归检查, 期望仍 100% (改编码不应伤害既有能力)
  4. nosizes_new   No-Sizes (无长度无方向),   新编码 -> 重测防御天花板 (旧值 70.59%)

超参与 run_quic_vs_masque_bi.py / run_ablations_bi.py 严格一致。
fine-tune.py 训练结束会自动在 data-test.json 上评测并写 test_stats.json。

    python experiments/run_iat_encoding_fix.py            # 训练 4 组 + 汇总
    python experiments/run_iat_encoding_fix.py --summary  # 只汇总已有结果
"""

import argparse
import json
import os
import subprocess
import sys
import time

# 仓库根:优先取 MASQUE_ROOT 环境变量,否则由本文件位置推导。
_ROOT = os.environ.get("MASQUE_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BYTE_PACKET_NUM = 5
SEQ_PACKET_NUM = 20
HEADER_BYTE_NUM = 80
PAYLOAD_BYTE_NUM = 240
STRIDE_LEN = 4

# (条件 id, 数据集目录, 输出目录, --interval_encoding, 汇报用名称, 旧编码下的既有数字)
CONDITIONS = [
    ("nsnb_old",    "bi_abl_no_sizes_nobytes", "iat_enc_nsnb_old",    "sigmoid_log",
     "No-Sizes+No-Bytes (旧编码, 复现对照)", "52.94%"),
    ("nsnb_new",    "bi_abl_no_sizes_nobytes", "iat_enc_nsnb_new",    "log10_decade",
     "No-Sizes+No-Bytes (新编码) ★主命题",   "52.94%"),
    ("full_new",    "quic_vs_masque_bi_feat",  "iat_enc_full_new",    "log10_decade",
     "Full 全特征基线 (新编码, 回归检查)",   "100.00%"),
    ("nosizes_new", "bi_nosizes",              "iat_enc_nosizes_new", "log10_decade",
     "No-Sizes 无长度无方向 (新编码)",       "70.59%"),
]

# 参照用: RF 只用 IAT、同条件 (first20, 同一会话级切分) 的结果
RF_REFERENCE = {"test": 1.0000, "cv_mean": 0.9683, "cv_std": 0.0276}


def train(data_name, out_name, encoding, tag, epochs=30, cuda_idx=0):
    data_dir = os.path.join(_ROOT, "data", "processed", data_name)
    output_dir = os.path.join(_ROOT, "outputs", out_name)
    ckpt_path = os.path.join(_ROOT, "NetMambaPlus", "dataset", "checkpoints",
                             "pre-trained checkpoints", "fuse3_mamba.pth")
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(data_dir, "metadata.json"), encoding="utf-8") as f:
        num_class = len(json.load(f)["name_to_idx"])

    print(f"\n{'='*74}\n训练条件: {tag}\n  data     = {data_dir}\n"
          f"  encoding = {encoding}\n  output   = {output_dir}\n{'='*74}", flush=True)

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
    --interval_encoding {encoding} \\
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
    ap.add_argument("--only", type=str, default=None,
                    help="只跑指定条件 id (逗号分隔)")
    args = ap.parse_args()

    conds = CONDITIONS
    if args.only:
        want = {s.strip() for s in args.only.split(",")}
        conds = [c for c in CONDITIONS if c[0] in want]

    if not args.summary:
        for cid, data_name, out_name, enc, tag, _ in conds:
            train(data_name, out_name, enc, tag, epochs=args.epochs)

    rows = []
    for cid, data_name, out_name, enc, tag, old in CONDITIONS:
        try:
            s = read_stats(out_name)
        except FileNotFoundError:
            print(f"(跳过未训练的条件: {cid})")
            continue
        rows.append({"id": cid, "tag": tag, "data": data_name, "encoding": enc,
                     "prev_number": old, "acc": s["acc"], "weighted_f1": s["weighted_f1"],
                     "cm": s["cm"]})

    out = os.path.join(_ROOT, "outputs", "summary_iat_encoding_fix.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"conditions": rows, "rf_reference_iat_only": RF_REFERENCE}, f,
                  indent=2, ensure_ascii=False)

    print(f"\n\n{'='*96}")
    print("U2 IAT 编码修复 (双向 + 会话级切分 + signed_sizes, n=51 test sessions)")
    print(f"{'='*96}")
    print(f"{'条件':<40}{'编码':<14}{'旧数字':>9}{'本次':>10}{'F1':>9}   混淆矩阵")
    print("-" * 96)
    for r in rows:
        print(f"{r['tag']:<38}{r['encoding']:<14}{r['prev_number']:>9}"
              f"{r['acc']*100:>9.2f}%{r['weighted_f1']:>9.4f}   {r['cm']}")
    print("-" * 96)
    print(f"参照 (只用 IAT 的随机森林, 同条件): test {RF_REFERENCE['test']*100:.2f}% / "
          f"5-fold {RF_REFERENCE['cv_mean']*100:.2f}% +- {RF_REFERENCE['cv_std']*100:.2f}% (n=504)")

    d = {r["id"]: r["acc"] for r in rows}
    if "nsnb_old" in d and "nsnb_new" in d:
        print(f"\n主命题 (只剩时间通道): {d['nsnb_old']*100:.2f}% -> {d['nsnb_new']*100:.2f}% "
              f"({(d['nsnb_new']-d['nsnb_old'])*100:+.2f} pp)")
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
