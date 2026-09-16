"""[2026-08-23] E3 作图: OOF accuracy + Wilson 95% CI, 以及方差来源分解。

论文里所有裸精度数字都要换成带区间的版本 (E3_handoff.md §12), 这张图就是那个替换的可视化。
CLAUDE.md 规则 2: matplotlib.use('Agg'), 缺中文字体 -> 图内一律英文。
CLAUDE.md 规则 4: 输出 docs/04_Meetings_and_Weekly_Notes/plots/, dpi=150。

    python experiments/plot_e3_kfold.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.environ.get("MASQUE_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY = os.path.join(_ROOT, "outputs", "e3_kfold", "summary_e3_kfold.json")
PLOT_DIR = os.path.join(_ROOT, "docs", "04_Meetings_and_Weekly_Notes", "plots")

# 图内一律英文 (环境缺中文字体)
LABELS = [
    ("full",            "Full (all features)"),
    ("no_intervals",    "No-Intervals"),
    ("c2",              "C2  CID normalised"),
    ("no_direction",    "No-Direction"),
    ("no_bytes",        "No-Bytes"),
    ("c1",              "C1  MTU normalised"),
    ("c3",              "C3  Retry removed"),
    ("c123",            "C123  all 3 constants"),
    ("c123_nobytes",    "C123 + No-Bytes"),
    ("no_sizes",        "No-Sizes"),
    ("nosizes_nobytes", "No-Sizes + No-Bytes"),
]
MAJORITY = 53.57


def main():
    with open(SUMMARY, encoding="utf-8") as f:
        S = json.load(f)
    C = S["conditions"]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                  gridspec_kw={"width_ratios": [1.55, 1]})

    # ---------------- 左: OOF accuracy + Wilson 95% CI ----------------
    rows = [(lab, C[cid]) for cid, lab in LABELS if cid in C]
    rows.sort(key=lambda r: r[1]["acc"])
    ys = range(len(rows))
    accs = [r[1]["acc"] * 100 for r in rows]
    lo = [a - r[1]["ci95"][0] * 100 for a, r in zip(accs, rows)]
    hi = [r[1]["ci95"][1] * 100 - a for a, r in zip(accs, rows)]
    colors = ["tomato" if r[0].startswith("Full") else "cornflowerblue" for r in rows]

    ax.errorbar(accs, ys, xerr=[lo, hi], fmt="none", ecolor="0.35",
                elinewidth=1.4, capsize=4, zorder=2)
    ax.scatter(accs, ys, s=64, c=colors, zorder=3, edgecolors="white", linewidths=0.8)
    for y, a, r in zip(ys, accs, rows):
        ax.text(a, y + 0.26, f"{a:.2f}%  ({r[1]['n'] - r[1]['correct']} err)",
                ha="center", va="bottom", fontsize=8.5, color="0.25")

    ax.set_yticks(list(ys))
    ax.set_yticklabels([r[0] for r in rows], fontsize=10)
    ax.set_xlabel("Out-of-fold accuracy (%), n = 504 flows = 504 sessions", fontsize=10)
    ax.set_title("E3  5-fold out-of-fold accuracy with Wilson 95% CI\n"
                 "(replaces the single 51-flow test set: 1 error = 1.96 pp)",
                 fontsize=11.5)
    ax.set_xlim(90, 101.4)
    ax.set_ylim(-1.15, len(rows) - 0.35)
    ax.grid(axis="x", alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)
    ax.axvline(97.82, color="darkorange", linestyle="--", linewidth=1.2, zorder=1)
    ax.text(97.72, -0.95, "11 mislabelled flows (493/504 = 97.82%) ",
            fontsize=8, color="darkorange", ha="right", va="center")
    ax.text(0.985, 0.02, f"majority-class baseline = {MAJORITY:.2f}%  (off-scale)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="0.45")

    # ---------------- 右: 方差来源分解 (Tier C) ----------------
    tierC = [("full", "Full"), ("c123", "C123"), ("nosizes_nobytes", "No-Sizes\n+No-Bytes")]
    tierC = [(c, l) for c, l in tierC if c in C and f"{c}_s1" in C]
    xs = range(len(tierC))
    fold_std = [C[c]["fold_std"] * 100 for c, _ in tierC]
    seed_range = []
    for c, _ in tierC:
        a = [C[k]["acc"] * 100 for k in (c, f"{c}_s1", f"{c}_s2")]
        seed_range.append(max(a) - min(a))

    w = 0.36
    b1 = ax2.bar([x - w / 2 for x in xs], fold_std, w, color="cornflowerblue",
                 label="Fold-to-fold std\n(= changing the test flows)")
    b2 = ax2.bar([x + w / 2 for x in xs], seed_range, w, color="tomato",
                 label="Seed-to-seed range\n(= training randomness, 3 seeds)")
    for bars in (b1, b2):
        for b in bars:
            ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.08,
                     f"{b.get_height():.2f}", ha="center", fontsize=9, color="0.25")

    ax2.set_xticks(list(xs))
    ax2.set_xticklabels([l for _, l in tierC], fontsize=9.5)
    ax2.set_ylabel("Accuracy spread (percentage points)", fontsize=10)
    ax2.set_title("Where the spread comes from\n(Tier C: 3 conditions x 3 seeds)", fontsize=11.5)
    ax2.legend(fontsize=8.5, loc="upper left", frameon=False)
    ax2.grid(axis="y", alpha=0.3, linestyle=":")
    ax2.set_axisbelow(True)
    ax2.set_ylim(0, max(fold_std + seed_range) * 1.45)

    fig.tight_layout(rect=(0, 0.055, 1, 1))
    fig.text(0.5, 0.022,
             "Caveat: 504 flows = 504 sessions (the group constraint is vacuous); "
             "site overlap across folds is 86.3% — this is NOT cross-site generalisation",
             ha="center", fontsize=8.5, color="0.4")

    os.makedirs(PLOT_DIR, exist_ok=True)
    out = os.path.join(PLOT_DIR, "e3_kfold_oof_ci.png")
    fig.savefig(out, dpi=150)
    print(f"已保存 {out}")


if __name__ == "__main__":
    main()
