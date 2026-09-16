# -*- coding: utf-8 -*-
"""E1 / 报头几何分析 组会用图（英文标注、dpi=150）。

四张图，对应 build_meeting_20260831_EN.py 的 slide 3/4/5/6：

  slide_e1_pa_law.png        P-A: min packet size = |DCID| + 23, holds for BOTH classes
  slide_e1_rules.png         each constant alone is a near-perfect zero-training rule
  slide_e1_ablation.png      C1/C2/C3/C123 forest plot (E3 pooled OOF + Wilson CI)
  slide_e1_verification.png  three checks that the erasure really bit

数据来源（全部实测，不硬编码推测值）：
  outputs/header_geometry/summary_header_geometry.json   P-A / P-B / P-C / zero-training rules
  outputs/e3_kfold/*/fold*/preds.json                    pooled OOF accuracy -> Wilson CI
  outputs/e3_kfold/mcnemar_full_matrix.json              exact McNemar, Holm over 10 comparisons
  outputs/const_rf_baseline/summary_const_rf_baseline.json  length-channel RF, rule collapse

    wsl -e bash -lic 'conda activate MAMBA; cd /mnt/d/zemi/2026/dataset; \
        python experiments/plot_e1_constants.py'
"""
import glob
import io
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.environ.get("MASQUE_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
OUT = os.path.join(_ROOT, "docs", "04_Meetings_and_Weekly_Notes", "plots")

DIRECT = "cornflowerblue"
MASQUE = "tomato"
NAVY = "#1F4E79"
ACCENT = "#2E75B6"
MUTED = "#777777"
GOLD = "#C0392B"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#999999",
    "axes.labelcolor": "#2D2D2D",
    "text.color": "#2D2D2D",
    "xtick.color": "#555555",
    "ytick.color": "#555555",
})


def load(path):
    with io.open(os.path.join(_ROOT, path), encoding="utf-8") as f:
        return json.load(f)


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def oof(cond):
    """Pool the 5 folds' per-sample predictions -> (correct, n)."""
    pat = os.path.join(_ROOT, "outputs", "e3_kfold", cond, "fold*", "preds.json")
    ok = tot = 0
    for fp in sorted(glob.glob(pat)):
        with io.open(fp, encoding="utf-8") as f:
            for _, rec in json.load(f).items():
                tot += 1
                ok += int(rec["pred"] == rec["label"])
    if tot != 504:
        raise SystemExit("%s: pooled %d flows, expected 504" % (cond, tot))
    return ok, tot


def finish(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path)


# --------------------------------------------------------------------------
# 1. P-A : smallest packet in a direction == |DCID| + 23, for BOTH classes
# --------------------------------------------------------------------------
def plot_pa():
    geo = load("outputs/header_geometry/summary_header_geometry.json")
    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    lo, hi = 18, 48
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1.2, ls="--", zorder=1,
            label="prediction  min = |DCID| + 23")

    # Several groups share a predicted value (and Direct/Masque groups overlap
    # exactly at |DCID| = 0 and 8 -- that coincidence is the point). Label the
    # x axis by group instead of annotating each marker, or the text collides.
    seen = set()
    for g in sorted(geo["P_A"], key=lambda g: -g["flows"]):
        c = DIRECT if g["cls"] == "Direct" else MASQUE
        lab = None
        if g["cls"] not in seen:
            lab = "%s  (area $\\propto$ flows)" % g["cls"]
            seen.add(g["cls"])
        ax.scatter(g["pred"], g["median_min"], s=30 + 1.7 * g["flows"],
                   color=c, alpha=0.75, edgecolor="white", lw=1.4,
                   zorder=3, label=lab)

    # |DCID| = 14/17/20 map to predictions 37/40/43, too close to label per
    # tick without collisions -- the range is stated in the callout instead.
    ax.set_xticks([20, 25, 30, 35, 40, 45])
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Predicted from RFC 9000:  |DCID| + 23  (bytes)",
                  fontsize=11, labelpad=8)
    ax.set_ylabel("Measured median minimum packet (bytes)", fontsize=11)
    ax.set_title("The law is a property of QUIC, not of MASQUE",
                 fontsize=13, color=NAVY, fontweight="bold", pad=12)
    resid = max(abs(g["resid"]) for g in geo["P_A"])
    ax.text(0.03, 0.955,
            "9 groups,  |DCID| $\\in$ {0, 8, 14, 17, 20},  n = 504 flows\n"
            "largest residual = %d B   (no fitting, no model)" % resid,
            transform=ax.transAxes, fontsize=10, va="top", color=NAVY,
            bbox=dict(boxstyle="round,pad=0.45", fc="#EBF3FA", ec=ACCENT, lw=1.1))
    ax.legend(loc="lower right", fontsize=9.5, frameon=False)
    ax.text(0.975, 0.30, "one marker = one (class, direction, |DCID|) group",
            transform=ax.transAxes, fontsize=8.5, color=MUTED, ha="right")
    ax.grid(alpha=0.25, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    finish(fig, "slide_e1_pa_law.png")


# --------------------------------------------------------------------------
# 2. Zero-training rules: each constant alone is a near-perfect detector
# --------------------------------------------------------------------------
def plot_rules():
    geo = load("outputs/header_geometry/summary_header_geometry.json")
    r = geo["rules"]
    items = [
        ("has Retry\n(single plaintext boolean)", r["has_retry"]),
        ("max packet == 1200 B\n(P-B)", r["max_eq_1200"]),
        ("min packet >= 42 B\n(P-A)", r["min_ge_42"]),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ys = range(len(items))
    accs = [v["acc"] * 100 for _, v in items]
    bars = ax.barh(list(ys), accs, height=0.52, color=ACCENT, alpha=0.90,
                   edgecolor="white", zorder=3)
    bars[0].set_color(GOLD)

    for y, (_, v) in zip(ys, items):
        ax.text(v["acc"] * 100 + 1.0, y,
                "%.2f%%    TPR %.1f%%  /  FPR %.1f%%"
                % (v["acc"] * 100, v["tpr"] * 100, v["fpr"] * 100),
                va="center", fontsize=10.5, color="#2D2D2D")

    base = 270 / 504 * 100
    ax.axvline(base, color=MUTED, lw=1.3, ls="--", zorder=4)
    ax.text(base + 1.0, -0.68, "majority class  %.2f%%" % base,
            fontsize=9.5, color=MUTED, va="top")

    ax.set_yticks(list(ys))
    ax.set_yticklabels([k for k, _ in items], fontsize=10.5)
    ax.set_ylim(len(items) - 0.35, -0.75)
    ax.set_xlim(45, 116)
    ax.set_xticks([50, 60, 70, 80, 90, 100])
    ax.set_xlabel("Accuracy on all 504 flows  (no training, no fitted threshold)",
                  fontsize=11, labelpad=8)
    ax.set_title("Each constant alone nearly solves the task",
                 fontsize=13, color=NAVY, fontweight="bold", pad=12)
    ax.grid(axis="x", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    finish(fig, "slide_e1_rules.png")


# --------------------------------------------------------------------------
# 3. E1 forest plot: erasing the constants costs no detectable accuracy
# --------------------------------------------------------------------------
def plot_ablation():
    mc = {c["B"]: c for c in load("outputs/e3_kfold/mcnemar_full_matrix.json")["comparisons"]}
    rows = [
        ("Full baseline", "full", "—"),
        ("C1  max datagram size", "c1", None),
        ("C2  connection-ID length", "c2", None),
        ("C3  Retry exchange", "c3", None),
        ("C123  all three", "c123", None),
        ("C123 + No-Bytes", "c123_nobytes", None),
    ]
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    fig.subplots_adjust(left=0.22, right=0.58, top=0.86, bottom=0.20)
    ys = list(range(len(rows)))[::-1]
    tx = ax.get_yaxis_transform()  # x in axes fraction, y in data coords

    for y, (lab, cond, ptxt) in zip(ys, rows):
        k, n = oof(cond)
        acc = k / n * 100
        lo, hi = [x * 100 for x in wilson(k, n)]
        focal = cond == "c123"
        col = GOLD if focal else (NAVY if cond == "full" else ACCENT)
        ax.plot([lo, hi], [y, y], color=col, lw=2.6 if focal else 2.0,
                alpha=0.85, zorder=3, solid_capstyle="round")
        ax.plot([acc], [y], "o", ms=11 if focal else 8, color=col,
                markeredgecolor="white", markeredgewidth=1.4, zorder=4)
        p = ptxt if ptxt else "p = %.2f" % mc[cond]["p_holm"]
        ax.text(1.04, y, "%.2f  [%.2f, %.2f]" % (acc, lo, hi), transform=tx,
                va="center", fontsize=10.5, color="#2D2D2D",
                fontweight="bold" if focal else "normal")
        ax.text(1.62, y, p, transform=tx, va="center", fontsize=10.5,
                color=GOLD if focal else MUTED,
                fontweight="bold" if focal else "normal")

    full_k, full_n = oof("full")
    ax.axvline(full_k / full_n * 100, color=NAVY, lw=1.0, ls=":", zorder=2)

    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=11)
    ax.set_xlim(93.5, 100.4)
    ax.set_ylim(-0.7, len(rows) - 0.28)
    ax.set_xlabel("Pooled out-of-fold accuracy, %   (n = 504, stratified 5-fold)",
                  fontsize=11, labelpad=7)
    ax.set_title("No condition differs significantly from the full baseline",
                 fontsize=13, color=NAVY, fontweight="bold", pad=14)
    ax.text(1.04, len(rows) - 0.42, "accuracy   [95% CI]", transform=tx,
            fontsize=9.5, color=MUTED, va="center")
    ax.text(1.62, len(rows) - 0.42, "McNemar", transform=tx,
            fontsize=9.5, color=MUTED, va="center")
    fig.text(0.055, 0.04,
             "bars = Wilson 95% CI;   p = exact McNemar vs. full baseline, "
             "Holm-corrected over 10 comparisons",
             fontsize=9, color=MUTED)
    ax.grid(axis="x", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    finish(fig, "slide_e1_ablation.png")


# --------------------------------------------------------------------------
# 4. Three checks: the erasure really bit, and the length channel is untouched
# --------------------------------------------------------------------------
def plot_verification():
    rf = load("outputs/const_rf_baseline/summary_const_rf_baseline.json")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.8, 4.6),
                                 gridspec_kw={"wspace": 0.30})
    fig.subplots_adjust(bottom=0.20, top=0.88)

    # --- left: the zero-training rules collapse to the majority class ---
    # summary JSON keys are Chinese; index by insertion order instead.
    # order = [max == 1242, min >= 84, contains Retry frame]; only the first two
    # are erasure checks for C123 (the Retry rule is read from the frame index,
    # not from the transformed features), so we plot those two -- same pair as
    # PROJECT_LOG table 4 and the paper.
    rules_orig = list(rf["part_b_unfitted_rules"]["orig"].values())[:2]
    rules_c123 = list(rf["part_b_unfitted_rules"]["C123"].values())[:2]
    labels = ["max packet\nrule", "min packet\nrule"]
    orig = [r["acc"] for r in rules_orig]
    after = [r["acc"] for r in rules_c123]
    base_maj = rf["part_a_three_constants"]["majority_baseline"]
    xs = range(len(labels))
    w = 0.34
    a1.bar([x - w / 2 for x in xs], orig, w, label="original data",
           color=ACCENT, edgecolor="white", zorder=3)
    a1.bar([x + w / 2 for x in xs], after, w, label="after C123",
           color="#B0B7BF", edgecolor="white", zorder=3)
    for x, v in zip(xs, orig):
        a1.text(x - w / 2, v + 1.4, "%.2f" % v, ha="center", fontsize=10, color=ACCENT)
    # the grey bars land exactly on the majority-class line, so one shared
    # label carries both numbers -- annotating each bar as well collides.
    for x, r in enumerate(rules_c123):
        a1.text(x + w / 2, after[x] / 2, "TPR\n%.1f%%" % r["tpr"], ha="center",
                va="center", fontsize=9.5, color="#3D4247")
    a1.axhline(base_maj, color=MUTED, lw=1.2, ls="--", zorder=4)
    a1.text(0.5, base_maj + 2.4, "majority class  %.2f%%" % base_maj,
            fontsize=9.5, color="#5A6068", ha="center")
    a1.set_xticks(list(xs))
    a1.set_xticklabels(labels, fontsize=10.5)
    a1.set_ylim(0, 128)
    a1.set_ylabel("Accuracy, %  (n = 504)", fontsize=10.5)
    a1.set_title("Checks 1-2:  the constants really are gone",
                 fontsize=11.5, color=NAVY, fontweight="bold", pad=10)
    a1.legend(fontsize=9.5, frameon=False, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, 1.005), columnspacing=1.2, handlelength=1.4)
    a1.grid(axis="y", alpha=0.25, lw=0.6)
    a1.set_axisbelow(True)
    for s in ("top", "right"):
        a1.spines[s].set_visible(False)

    # --- right: length-channel RF is untouched ---
    # keys are Chinese; insertion order is orig / C1 / C2 / C3 / C123
    order = ["orig", "C1", "C2", "C3", "C123"]
    entries = list(rf["part_c_length_channel_rf"].values())
    if len(entries) != len(order):
        raise SystemExit("part_c_length_channel_rf: expected 5 entries")
    accs = [e["cv_mean"] for e in entries]
    errs = [e["cv_std"] for e in entries]
    cols = [MUTED] + [ACCENT] * 3 + [GOLD]
    a2.bar(range(len(order)), accs, 0.56, yerr=errs, capsize=4,
           color=cols, edgecolor="white", zorder=3,
           error_kw=dict(ecolor="#666666", lw=1.1))
    for x, (v, e) in enumerate(zip(accs, errs)):
        a2.text(x, v + e + 0.30, "%.2f" % v, ha="center", fontsize=10,
                color="#2D2D2D",
                fontweight="bold" if order[x] in ("orig", "C123") else "normal")
    a2.set_xticks(range(len(order)))
    a2.set_xticklabels(order, fontsize=10.5)
    a2.set_ylim(96.0, 101.9)
    a2.set_ylabel("5-fold accuracy, %  (n = 504)", fontsize=10.5)
    a2.set_title("Check 3:  length channel is untouched",
                 fontsize=11.5, color=NAVY, fontweight="bold", pad=10)
    a2.annotate("", xy=(4, 101.05), xytext=(0, 101.05),
                arrowprops=dict(arrowstyle="<->", color=GOLD, lw=1.4))
    a2.text(2, 101.20, "%.2f  $\\rightarrow$  %.2f" % (accs[0], accs[-1]),
            ha="center", fontsize=11, color=GOLD, fontweight="bold")
    a2.grid(axis="y", alpha=0.25, lw=0.6)
    a2.set_axisbelow(True)
    for s in ("top", "right"):
        a2.spines[s].set_visible(False)

    fig.text(0.52, 0.035,
             "left: zero-training threshold rules   |   "
             "right: random forest on 24 packet-length summary statistics",
             fontsize=9, color=MUTED, ha="center")
    finish(fig, "slide_e1_verification.png")


if __name__ == "__main__":
    plot_pa()
    plot_rules()
    plot_ablation()
    plot_verification()
