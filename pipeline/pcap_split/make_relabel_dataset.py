"""[2026-09-08] E6-F: 把 11 条标签噪声流【改正标签】后重建数据集。

## 为什么"排除"不够, 要"改正 + 重训"

论文的处理是: 保留这 11 条流不动, 然后在 Limitations 里说明"全模型 10 个错误里
有 9 个是它们"。这在两头占便宜 —— 读者会自动把 98.02% 心算成 99.8%, 但数据没改。
而 outputs/e3_kfold/error_analysis_e3.json 里"排除这 11 条"算出的 99.80% 也只是
【评测侧】的修正: 模型训练时看到的仍然是错标签, 11 条实际为 direct 的流被当成
MASQUE 教给了模型。

这 11 条的身份是【独立于模型预测】判定的 —— 它们的对端 IP 不在 WARP 的
162.159 ingress 前缀内, 也就是抓在隧道断开窗口里的真实直连流量。所以把它们的
标签改成 Direct 不是"把模型做错的样本挪走"(那才是拿数据拟合结果), 而是修正
ground truth。改正后重训一遍, 才能回答"这个数据集的真实上限是多少"。

    python pipeline/pcap_split/make_relabel_dataset.py
"""

import copy
import json
import os

_ROOT = os.environ.get("MASQUE_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROC = os.path.join(_ROOT, "data", "processed")
SRC = "quic_vs_masque_bi_feat"
DST = "e6_relabel"
ERR = os.path.join(_ROOT, "outputs", "e3_kfold", "error_analysis_e3.json")
SPLITS = ["train", "valid", "test"]


def main():
    with open(ERR, encoding="utf-8") as f:
        noise = set(json.load(f)["label_noise_flows"])
    print(f"标签噪声流: {len(noise)} 条 (判定依据: 对端不在 WARP 162.159 前缀内)")

    with open(os.path.join(PROC, SRC, "metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    name_to_idx = meta["name_to_idx"]
    direct_name = next(n for n, i in name_to_idx.items() if i == 0)
    print(f"目标标签: {direct_name} -> {name_to_idx[direct_name]}")

    os.makedirs(os.path.join(PROC, DST), exist_ok=True)
    hit, counts = 0, {}
    for sp in SPLITS:
        with open(os.path.join(PROC, SRC, f"data-{sp}.json"), encoding="utf-8") as f:
            items = json.load(f)
        out = []
        for it in items:
            c = copy.deepcopy(it)
            if os.path.basename(it["pcap_file"]) in noise:
                assert c["label"] == 1, f"噪声流标签本应为 1: {it['pcap_file']}"
                c["label"] = name_to_idx[direct_name]
                c["name"] = direct_name
                c["relabelled"] = True
                hit += 1
            out.append(c)
            counts[c["label"]] = counts.get(c["label"], 0) + 1
        with open(os.path.join(PROC, DST, f"data-{sp}.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"  data-{sp}.json: {len(out)} 条")

    with open(os.path.join(PROC, DST, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    assert hit == len(noise), f"只改到 {hit}/{len(noise)} 条, 文件名对不上"
    total = sum(counts.values())
    maj = max(counts.values()) / total
    print(f"\n改正 {hit} 条。新的类别分布: "
          + " / ".join(f"label{k} {v}" for k, v in sorted(counts.items())))
    print(f"新的多数类基线: {maj*100:.2f}%  (原 53.57%)")
    print(f"\n⚠️ 折分配沿用 outputs/e3_kfold/fold_assignment.json (样本集合未变, 仅标签变),")
    print(f"   故与其他条件仍严格配对; 但分层是按原标签算的, 各折的类别比例会略有偏移。")


if __name__ == "__main__":
    main()
