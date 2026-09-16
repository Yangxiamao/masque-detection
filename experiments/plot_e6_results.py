# -*- coding: utf-8 -*-
"""[2026-09-08] E6 两张主图: 单通道充分性 / timing 在观测点抖动下的衰减。

配色与既有图一致 (CLAUDE.md 规则 4): Direct = cornflowerblue, MASQUE = tomato,
dpi=150, 全英文标注 (环境缺中文字体)。

数字全部从 outputs 下的 JSON 读取, 脚本本身不写死任何精度值。

    wsl -e bash -lic 'conda activate MAMBA; cd /mnt/d/zemi/2026/dataset;
                      python experiments/plot_e6_results.py'
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_ROOT = os.environ.get("MASQUE_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(__file__), ".."))
E3 = os.path.join(_ROOT, "outputs", "e3_kfold")
PLOTS = os.path.join(_ROOT, "docs", "04_Meetings_and_Weekly_Notes", "plots")

BLUE, RED, GREY = "cornflowerblue", "tomato", "0.55"
MAJORITY = 270 / 504 * 100


def oof_acc(cond):
    """由逐折 preds.json 汇总 OOF accuracy; 缺条件返回 None。"""
    oof = {}
    for k in range(5):
        p = os.path.join(E3, cond, f"fold{k}", "preds.json")
        if not os.path.exists(p):
            return None
        with open(p, encoding="utf-8") as f:
            oof.update(json.load(f))
    if not oof:
        return None
    return sum(v["pred"] == v["label"] for v in oof.values()) / len(oof) * 100


def panel_sufficiency(ax):
    """每个通道单独训练能到多少 —— 论文 Table I 无法分离 |len| 与 dir, 这里补上。"""
    rows = [
        ("All channels",        "full",        GREY),
        ("Packet length only",  "suff_mag",    BLUE),
        ("Direction only",      "suff_dir",    RED),
        ("Timing only",         "nosizes_nobytes", "mediumseagreen"),
        ("Header bytes only",   "suff_bytes",  "goldenrod"),
    ]
    labels, vals, colors = [], [], []
    for lab, cond, col in rows:
        a = oof_acc(cond)
        if a is None:
            print(f"  (跳过未完成条件: {cond})")
            continue
        labels.append(lab)
        vals.append(a)
        colors.append(col)

    y = range(len(vals))
    ax.barh(list(y), vals, color=colors, edgecolor="black", linewidth=0.6, height=0.55)
    ax.axvline(MAJORITY, color="black", ls="--", lw=1.1)
    # 标注放到全图【顶部】新开的留白带里: 压到 x 轴刻度或压到柱子数值都不行
    ax.text(MAJORITY + 1.0, -0.78, f"majority class {MAJORITY:.2f}%",
            fontsize=8, va="center", ha="left")
    for i, v in enumerate(vals):
        ax.text(v - 1.2, i, f"{v:.2f}%", va="center", ha="right",
                fontsize=9, color="white", fontweight="bold")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_ylim(len(vals) - 0.35, -1.15)   # 顶部留白供基线标注; 同时完成 y 轴反转
    ax.set_xlim(50, 101)
    ax.set_xlabel("Pooled out-of-fold accuracy (%), n = 504 sessions", fontsize=9)
    ax.set_title("(a) Each channel trained in isolation", fontsize=10, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)


def panel_jitter(ax):
    """timing-only 在观测点抖动下的衰减 + 微秒 bin 重要度的塌陷。"""
    with open(os.path.join(_ROOT, "outputs", "e6_jitter",
                           "summary_e6_jitter_stats.json"), encoding="utf-8") as f:
        js = json.load(f)

    order = [("client (sigma=0)", 0.0, "nosizes_nobytes"),
             ("sigma=50us",  50.0,   "jit50us"),
             ("sigma=200us", 200.0,  "jit200us"),
             ("sigma=1ms",   1000.0, "jit1ms"),
             ("sigma=5ms",   5000.0, "jit5ms")]

    xs = list(range(len(order)))
    rf = [js[tag]["models"]["RandomForest"]["cv_mean"] for tag, _, _ in order]
    rf_e = [js[tag]["models"]["RandomForest"]["cv_std"] for tag, _, _ in order]
    imp = [js[tag]["importance_sub100us"] * 100 for tag, _, _ in order]
    nm = [oof_acc(c) for _, _, c in order]

    ax.errorbar(xs, rf, yerr=rf_e, marker="o", color=BLUE, lw=1.8, capsize=3,
                label="Random forest, 19 IAT statistics")
    if all(v is not None for v in nm):
        ax.plot(xs, nm, marker="s", color=RED, lw=1.8,
                label="NetMamba+, timing channel only")
    else:
        done = [(x, v) for x, v in zip(xs, nm) if v is not None]
        if done:
            ax.plot([d[0] for d in done], [d[1] for d in done], marker="s",
                    color=RED, lw=1.8, label="NetMamba+, timing channel only (partial)")
    ax.axhline(MAJORITY, color="black", ls="--", lw=1.1)
    ax.text(2.0, MAJORITY + 1.4, f"majority class {MAJORITY:.2f}%", fontsize=8,
            ha="center")

    ax.set_xticks(xs)
    ax.set_xticklabels(["client\n(no jitter)", "50 $\\mu$s", "200 $\\mu$s",
                        "1 ms", "5 ms"], fontsize=8.5)
    ax.set_xlabel("Gaussian jitter added at the observation point ($\\sigma$)", fontsize=9)
    ax.set_ylabel("Accuracy (%)", fontsize=9)
    ax.set_ylim(45, 102)
    ax.set_title("(b) Timing alone, moving the observer off-host",
                 fontsize=10, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)

    ax2 = ax.twinx()
    ax2.bar(xs, imp, color=GREY, alpha=0.30, width=0.5, zorder=0)
    ax2.set_ylabel("RF importance of sub-100 $\\mu$s bins (%)", fontsize=8.5, color="0.35")
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis="y", labelsize=8, colors="0.35")

    # 两条曲线都从左上向右下走, 右上角是空的; 放左下会压住 sigma=0 的灰柱与基线标注
    h1, l1 = ax.get_legend_handles_labels()
    grey = matplotlib.patches.Patch(facecolor=GREY, alpha=0.30,
                                    label="sub-100 $\\mu$s importance (right axis)")
    ax.legend(h1 + [grey], l1 + [grey.get_label()], fontsize=7.5,
              loc="upper right", framealpha=0.92)


def main():
    os.makedirs(PLOTS, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.9))
    panel_sufficiency(axes[0])
    panel_jitter(axes[1])
    fig.tight_layout(pad=1.4)
    out = os.path.join(PLOTS, "e6_sufficiency_and_jitter.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
