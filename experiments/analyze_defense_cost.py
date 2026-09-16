# -*- coding: utf-8 -*-
"""[2026-09-15] 量化两种模拟防御在被分析窗口上的代价 (审稿意见: 删除或量化 "padding is near-free")。

## 动机

论文 V-E 原来写 "padding is near-free here, since MASQUE's packets already cluster
near the MTU"。教授指出 Fig. 2 里 MASQUE 前 50 包窗口内小包极多, 把它们全部填到
1350 B 的带宽开销未必小, 要求删除或量化。本脚本直接在训练用特征
(data/processed/quic_vs_masque_bi_feat/data.json, 每流前 50 包的 signed_sizes /
intervals) 上算:

  * padding 代价: sum(1350 * n) / sum(|size|), 按类汇总, 另给逐流中位数;
  * shaping 代价: 恒 5 ms 间隔下 50 包窗口时长 = 49 * 5 ms = 245 ms, 与实测窗口
    时长比较 —— 注意模拟变换把长间隔也压短了, 这是真实客户端做不到的
    (只能延后不能提前), 所以 shaping 的真实代价是延迟而非这里能算的量。

只覆盖被分析的 50 包窗口; 整段会话上 MASQUE 下行 77.6% 已是满包, 开销会更小,
但本脚本没有算 (需要回到 pcap)。

    python experiments/analyze_defense_cost.py
"""
import json
import os
import statistics as st

_ROOT = os.environ.get("MASQUE_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(_ROOT, "data", "processed", "quic_vs_masque_bi_feat", "data.json")
OUT = os.path.join(_ROOT, "outputs", "e6_defense_cost")

PAD = 1350        # simulate_obfuscation_bi.py 的 MTU padding 目标
GAP = 0.005       # interval shaping 的恒定间隔 (s)


def main():
    with open(DATA, encoding="utf-8") as f:
        rows = json.load(f)
    result = {"pad_bytes": PAD, "shape_gap_s": GAP, "window_packets": 50, "classes": {}}
    for cls in ("Direct", "Masque"):
        sub = [r for r in rows if r["name"] == cls]
        tot = padtot = 0
        ratios, durs = [], []
        for r in sub:
            sizes = [abs(int(x)) for x in r["signed_sizes"].split()]
            ivs = [float(x) for x in r["intervals"].split()]
            tot += sum(sizes)
            padtot += PAD * len(sizes)
            ratios.append(PAD * len(sizes) / sum(sizes))
            durs.append(sum(ivs))
        n_pkts = sum(len(r["signed_sizes"].split()) for r in sub)
        shaped = GAP * (50 - 1)
        result["classes"][cls] = {
            "n_flows": len(sub),
            "mean_abs_size_B": round(tot / n_pkts, 1),
            "padded_over_original_bytes": round(padtot / tot, 3),
            "per_flow_ratio_median": round(st.median(ratios), 3),
            "window_duration_median_ms": round(st.median(durs) * 1000, 1),
            "window_duration_mean_ms": round(st.mean(durs) * 1000, 1),
            "shaped_window_ms": round(shaped * 1000, 1),
            "flows_lengthened_by_shaping": sum(1 for d in durs if shaped > d),
        }
        c = result["classes"][cls]
        print(f"{cls}: n={c['n_flows']} mean|size|={c['mean_abs_size_B']} B  "
              f"padded/original={c['padded_over_original_bytes']}x  "
              f"per-flow median={c['per_flow_ratio_median']}x  "
              f"window median={c['window_duration_median_ms']} ms vs shaped {c['shaped_window_ms']} ms")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "defense_cost.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("->", os.path.join(OUT, "defense_cost.json"))


if __name__ == "__main__":
    main()
