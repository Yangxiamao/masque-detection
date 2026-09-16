"""[2026-08-23] E3: 把某折的逐样本预测落盘, 供 McNemar 配对检验用。

为什么需要这个 (E3_handoff.md §4.1):
NetMambaPlus/src/engine_mm.py 的 evaluate() 收集了 pred_all/target_all, 但只返回聚合指标,
**没有把逐样本预测写盘, 也没有对应开关**。test_stats.json 里只有混淆矩阵 ——
**从混淆矩阵无法还原配对信息**, McNemar 做不了。

对齐依据 (已核实):
  util/loader_data.py:306-335  get_data_loader() 默认 random_sampler=False -> SequentialSampler,
                               drop_last=False; ratio=1.0 时 sample_data() 不打乱;
                               process_bi_flow_data() 原地改写, 保序; class_idx=None 不过滤
  models_net_mamba_fuse3.py:261  分类模式 if_mask=False -> 推理确定性, 无随机遮蔽
=> 预测顺序 == data-test.json 里记录的顺序, 直接 zip 对齐 pcap_file。

⚠️ 不改 NetMambaPlus/ 里的任何文件 (CLAUDE.md 规则 3, 它是 submodule)。
本脚本用 sys.path 引入它的模块。

    python experiments/eval_dump_preds.py \
        --data_dir data/processed/e3_folds/full/fold0 \
        --ckpt outputs/e3_kfold/full/fold0/checkpoint-best.pth \
        --out outputs/e3_kfold/full/fold0/preds.json
"""

import argparse
import json
import os
import sys

_ROOT = os.environ.get("MASQUE_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "NetMambaPlus", "src")
sys.path.insert(0, _SRC)

import torch  # noqa: E402

from util.arg_fine_tune import get_args_parser  # noqa: E402
from util.loader_data import get_data_loader  # noqa: E402
from util.loader_model import get_model_classifier  # noqa: E402
from engine_mm import get_data_processors, get_model_forward_fn  # noqa: E402

# 与 run_kfold_e3.py / run_const_ablation.py 严格一致的超参
HP = dict(num_packet=5, num_packet_byte=320, seq_len=20, stride_size=4,
          batch_size=16, model="fuse3_mamba_classifier",
          dataset_type="byte_size_interval", size_key="signed_sizes",
          interval_encoding="log10_decade")


def build_args(data_dir, nb_classes, seed, device):
    argv = [
        "--num_packet", str(HP["num_packet"]),
        "--num_packet_byte", str(HP["num_packet_byte"]),
        "--seq_len", str(HP["seq_len"]),
        "--stride_size", str(HP["stride_size"]),
        "--batch_size", str(HP["batch_size"]),
        "--nb_classes", str(nb_classes),
        "--data_path", data_dir,
        "--model", HP["model"],
        "--dataset_type", HP["dataset_type"],
        "--size_key", HP["size_key"],
        "--interval_encoding", HP["interval_encoding"],
        "--seed", str(seed),
        "--device", device,
        "--num_workers", "0",
        "--no_amp",
    ]
    return get_args_parser().parse_args(argv)


@torch.no_grad()
def dump(data_dir, ckpt_path, out_path, seed=0, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(data_dir, "metadata.json"), encoding="utf-8") as f:
        name_to_idx = json.load(f)["name_to_idx"]
    with open(os.path.join(data_dir, "data-test.json"), encoding="utf-8") as f:
        test_items = json.load(f)

    args = build_args(data_dir, len(name_to_idx), seed, device)

    model = get_model_classifier(args)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    msg = model.load_state_dict(ckpt["model"], strict=False)
    if msg.missing_keys:
        print(f"警告: 缺失权重 {msg.missing_keys}")
    if msg.unexpected_keys:
        print(f"警告: 多余权重 {msg.unexpected_keys}")
    model.to(device)
    model.eval()

    # 与 fine-tune.py:237-238 完全一致的调用形式 (未传 random_sampler -> SequentialSampler)
    loader, _ = get_data_loader(args, os.path.join(data_dir, "data-test.json"))
    process_data = get_data_processors()[args.dataset_type]
    forward_model = get_model_forward_fn()[args.dataset_type]

    preds, targets, probs = [], [], []
    for batch in loader:
        inputs = process_data(batch, device)
        with torch.cuda.amp.autocast():
            logits = forward_model(model, inputs)["logits"]
        _, p = logits.topk(1, 1, True, True)
        preds.extend(p.t()[0].cpu().tolist())
        targets.extend(inputs["targets"].cpu().tolist())
        # [2026-09-08 E6-E] 正类后验概率。准确率是阈值 0.5 上的一个点, 而论文
        # Introduction 设定的场景 (运营商在 UDP/443 上做策略执行) 里 MASQUE 的
        # 先验远低于本数据集的 53.6%, 那种 base rate 下真正要报的是低 FPR 工作点。
        # 没有概率就画不出 ROC, 而 checkpoint 跑完即删 -> 必须在这里落盘。
        probs.extend(logits.float().softmax(dim=1)[:, 1].cpu().tolist())

    if len(preds) != len(test_items):
        raise SystemExit(f"预测条数 {len(preds)} != data-test.json 条数 {len(test_items)}, 顺序假设失效")

    out = {}
    for item, p, t, pr in zip(test_items, preds, targets, probs):
        key = os.path.basename(item["pcap_file"])
        if item["label"] != t:
            raise SystemExit(f"标签错位: {key} json={item['label']} loader={t}, 顺序假设失效")
        out[key] = {"pred": int(p), "label": int(t), "prob1": float(pr)}

    acc = sum(v["pred"] == v["label"] for v in out.values()) / len(out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"dump {len(out)} 条 -> {out_path}   acc = {acc*100:.4f}%")
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="折数据集目录 (含 data-test.json / metadata.json)")
    ap.add_argument("--ckpt", required=True, help="checkpoint-best.pth 路径")
    ap.add_argument("--out", required=True, help="preds.json 输出路径")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--verify", default=None,
                    help="test_stats.json 路径; 给了就核对 acc 是否与其完全一致")
    args = ap.parse_args()

    absp = lambda p: p if os.path.isabs(p) else os.path.join(_ROOT, p)  # noqa: E731
    acc = dump(absp(args.data_dir), absp(args.ckpt), absp(args.out),
               seed=args.seed, device=args.device)

    if args.verify:
        with open(absp(args.verify), encoding="utf-8") as f:
            ref = json.load(f)["acc"]
        if abs(acc - ref) > 1e-9:
            raise SystemExit(f"!!! 不一致: dump acc {acc:.10f} vs test_stats {ref:.10f}")
        print(f"核对通过: dump acc == test_stats acc == {ref*100:.4f}%")


if __name__ == "__main__":
    main()
