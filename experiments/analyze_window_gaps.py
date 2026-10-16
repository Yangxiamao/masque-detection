# -*- coding: utf-8 -*-
"""[2026-10-16] 论文 IV-C "53% of the gaps in the 50-packet windows are below 100 us" 的出处。

在训练用特征 (data/processed/quic_vs_masque_bi_feat/data.json, 每流前 50 包的 intervals,
单位 s, 第 1 个恒为 0) 上统计包间隔 < 100 us 的比例: 50 包窗口 (每流 49 个间隔) 与
模型实际读的前 20 包 (每流 19 个间隔), 全体与按类。全体比例与标签无关;
按类比例随改标变化, 所以支持 MASQUE_RELABEL=1。

    python experiments/analyze_window_gaps.py
    MASQUE_RELABEL=1 python experiments/analyze_window_gaps.py   # -> outputs/window_gaps_relabel
"""
import json
import os
import sys as _sys

_ROOT = os.environ.get("MASQUE_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(_ROOT, "data", "processed", "quic_vs_masque_bi_feat", "data.json")
OUT = os.path.join(_ROOT, "outputs", "window_gaps")

_sys.path.insert(0, os.path.join(_ROOT, "pipeline"))
import relabel_util  # noqa: E402
OUT = relabel_util.out_dir(OUT)

THRESH = 1e-4     # 100 us


def count(rows, n_packets):
    below = total = 0
    for r in rows:
        gaps = [float(x) for x in r["intervals"].split()][1:n_packets]
        total += len(gaps)
        below += sum(g < THRESH for g in gaps)
    return {"gaps": total, "below_100us": below, "share": round(below / total, 4)}


def main():
    with open(DATA, encoding="utf-8") as f:
        rows = relabel_util.relabel_items(json.load(f))
    res = {"threshold_s": THRESH, "n_flows": len(rows)}
    for key, n in (("window_50", 50), ("first_20", 20)):
        res[key] = {"all": count(rows, n)}
        for cls in ("Direct", "Masque"):
            res[key][cls] = count([r for r in rows if r["name"] == cls], n)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=1))
    print("->", os.path.join(OUT, "summary.json"))


if __name__ == "__main__":
    main()
