# -*- coding: utf-8 -*-
"""[2026-09-15] 论文投稿版 (paper_conf) 攻防图: 各防御条件下 clean-trained 与两种
对抗训练检测器的 5 折 OOF 精度 —— Table adv 的图形化, 展示"崩溃-重训恢复-只有
关闭方向才守住"的模式。

数字全部从 outputs/e6_adv_kfold/summary_e6_adv_kfold.json 读取, 脚本不写死精度值。
单栏图 (3.5 in 宽), 300 dpi 以保证印刷清晰; 全英文标注 (环境缺中文字体)。

    wsl -e bash -lic 'conda activate MAMBA; cd /mnt/d/zemi/2026/dataset;
                      python experiments/plot_adv_defense_fig.py'
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_ROOT = os.environ.get("MASQUE_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(__file__), ".."))
SUMMARY = os.path.join(_ROOT, "outputs", "e6_adv_kfold", "summary_e6_adv_kfold.json")
PLOTS = os.path.join(_ROOT, "docs", "04_Meetings_and_Weekly_Notes", "plots")
OUT = os.path.join(PLOTS, "conf_adv_defense.png")

# 横轴: 施加到测试数据上的防御 (与 Table adv 的列一一对应)
CONDITIONS = [
    ("t1_clean",    "No defense\n(T1)"),
    ("t2_pad",      "Padding\n(T2)"),
    ("t3_padshape", "Padding\n+ shaping (T3)"),
    ("t3_absdir",   "T3 + direction\nclosed"),
]
# 每组三根柱: 检测器 (与 Table adv 的行一一对应)
DETECTORS = [
    ("clean",    "Clean-trained detector",          "0.72"),
    ("advpad",   "Adv. trained: padding",            "cornflowerblue"),
    ("advshape", "Adv. trained: padding + shaping",  "#1f4e79"),
]


def main():
    with open(SUMMARY, encoding="utf-8") as f:
        d = json.load(f)
    tab = d["table"]
    majority = d["majority"] * 100

    plt.rcParams.update({
        "font.size": 7.5, "axes.labelsize": 8, "legend.fontsize": 6.8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
    })
    fig, ax = plt.subplots(figsize=(3.5, 2.75), dpi=300)
    n_c, n_d = len(CONDITIONS), len(DETECTORS)
    width = 0.26
    x = np.arange(n_c)

    for j, (det, lab, col) in enumerate(DETECTORS):
        accs, lo, hi = [], [], []
        for cond, _ in CONDITIONS:
            r = tab[f"{det}|{cond}"]
            a = r["acc"] * 100
            accs.append(a)
            lo.append(a - r["ci95"][0] * 100)
            hi.append(r["ci95"][1] * 100 - a)
        xs = x + (j - (n_d - 1) / 2) * width
        ax.bar(xs, accs, width, color=col, edgecolor="white", linewidth=0.4,
               label=lab, zorder=3)
        ax.errorbar(xs, accs, yerr=[lo, hi], fmt="none", ecolor="0.25",
                    elinewidth=0.6, capsize=1.5, zorder=4)
        for xi, a, h in zip(xs, accs, hi):
            ax.text(xi, a + h + 0.8, f"{a:.1f}", ha="center", va="bottom",
                    fontsize=5.4, color="0.15")

    ax.axhline(majority, color="tomato", ls="--", lw=0.9, zorder=2,
               label=f"Majority class ({majority:.1f}%)")

    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in CONDITIONS])
    ax.set_ylim(40, 104)
    ax.set_yticks(range(40, 101, 10))
    ax.set_ylabel("Out-of-fold accuracy (%)")
    ax.set_xlim(-0.55, n_c - 0.45)
    ax.yaxis.grid(True, color="0.9", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    h, l = ax.get_legend_handles_labels()
    order = [1, 2, 3, 0]  # 三个检测器在前, majority 线最后
    fig.legend([h[i] for i in order], [l[i] for i in order],
               loc="lower center", frameon=False, ncol=2, handlelength=1.4,
               columnspacing=1.0, bbox_to_anchor=(0.55, 0.0))

    fig.tight_layout(pad=0.3, rect=(0, 0.12, 1, 1))
    fig.savefig(OUT)
    print("saved", OUT)
    # 回显所用数字, 便于与 Table adv 核对
    for det, lab, _ in DETECTORS:
        print(f"{lab:38s}", "  ".join(
            f"{tab[f'{det}|{c}']['acc']*100:6.2f}" for c, _ in CONDITIONS))


if __name__ == "__main__":
    main()
