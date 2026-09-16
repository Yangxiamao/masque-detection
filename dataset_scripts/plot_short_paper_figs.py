#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Column-width figures for the 2-page short paper (paper_short/masque_short.tex).

Why this exists: the figures in plot_bidirectional_analysis.py are drawn at
10-11 inches wide.  Shrinking them to a 3.5-inch IEEE column makes their tick
labels and annotations illegible (checked on the rendered PDF).  These are
built at their FINAL physical size, so a point size here is a point size on
the printed page.

Run in WSL conda MAMBA env:
  wsl -e bash -lic 'conda activate MAMBA; cd /mnt/d/zemi/2026/dataset/dataset_scripts; python plot_short_paper_figs.py'

Data: data/processed/quic_vs_masque_bi_feat/data.json
Out : figures/short_*.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")  # no display env
import matplotlib.pyplot as plt
import numpy as np

DIRECT_COLOR = "cornflowerblue"
MASQUE_COLOR = "tomato"
DPI = 300

# Final size on the page: one IEEE column is 3.5 in; we render at 3.35 in and
# include at width=\columnwidth so the mapping is ~1:1.
COL_W = 3.35

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.environ.get("MASQUE_ROOT") or os.path.dirname(SCRIPT_DIR)
DATA_FILE = os.path.join(_ROOT, "data", "processed",
                         "quic_vs_masque_bi_feat", "data.json")
OUT_DIR = os.path.join(_ROOT, "figures")

plt.rcParams.update({
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
})


def load_data():
    with open(DATA_FILE, "r") as f:
        data = json.load(f)
    n0 = sum(1 for r in data if r["label"] == 0)
    n1 = sum(1 for r in data if r["label"] == 1)
    print(f"Loaded {len(data)} flows  (Direct {n0} / MASQUE {n1})")
    return data


def parse_signed(rec):
    return [int(x) for x in rec["signed_sizes"].split()]


# ======================================================================
# Fig. 1 -- per-flow fingerprint heatmap, column width
# ======================================================================
def fig_fingerprint(data, n_pkt=40):
    direct_rows = [parse_signed(r)[:n_pkt] for r in data if r["label"] == 0]
    masque_rows = [parse_signed(r)[:n_pkt] for r in data if r["label"] == 1]

    def to_matrix(rows):
        M = np.full((len(rows), n_pkt), np.nan)
        for i, r in enumerate(rows):
            M[i, :len(r)] = r
        return M

    Md, Mm = to_matrix(direct_rows), to_matrix(masque_rows)
    M = np.vstack([Md, Mm])
    n_direct = len(direct_rows)

    fig, ax = plt.subplots(figsize=(COL_W, 1.65))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="lightgray")
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=-1400, vmax=1400,
                   interpolation="nearest")
    ax.axhline(n_direct - 0.5, color="black", linewidth=1.0)

    ax.set_xlabel("Packet index")
    ax.set_xticks(np.arange(0, n_pkt, 10))
    ax.set_xticklabels(np.arange(1, n_pkt + 1, 10))
    ax.set_yticks([n_direct / 2, n_direct + len(masque_rows) / 2])
    ax.set_yticklabels([f"Direct\n({n_direct})",
                        f"MASQUE\n({len(masque_rows)})"],
                       rotation=90, va="center")

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("signed size (+up / -down)", fontsize=6)
    cbar.ax.tick_params(labelsize=5.5, width=0.5)
    cbar.outline.set_linewidth(0.5)

    fig.tight_layout(pad=0.25)
    out = os.path.join(OUT_DIR, "short_flow_fingerprint.png")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"Saved: {out}")


# ======================================================================
# Fig. 2 -- bidirectional packet-size distribution, column width
# ======================================================================
def fig_size_distribution(data):
    up_d, up_m, dn_d, dn_m = [], [], [], []
    for rec in data:
        for v in parse_signed(rec):
            if v > 0:
                (up_d if rec["label"] == 0 else up_m).append(v)
            elif v < 0:
                (dn_d if rec["label"] == 0 else dn_m).append(-v)

    bins = np.linspace(0, 1600, 100)
    fig, (ax_up, ax_dn) = plt.subplots(2, 1, figsize=(COL_W, 2.05), sharex=True)

    def panel(ax, d, m, ylab):
        ax.hist(d, bins=bins, alpha=0.65, density=True, color=DIRECT_COLOR,
                label=f"Direct  n={len(d):,}", linewidth=0)
        ax.hist(m, bins=bins, alpha=0.65, density=True, color=MASQUE_COLOR,
                label=f"MASQUE  n={len(m):,}", linewidth=0)
        ax.set_ylabel(ylab, fontsize=7)
        ax.legend(loc="upper center", frameon=False, handlelength=1.0,
                  borderpad=0.2, labelspacing=0.2)
        ax.grid(axis="y", alpha=0.35, linestyle="--", linewidth=0.4)
        ax.tick_params(length=2.5)
        ax.set_yticks([])

    panel(ax_up, up_d, up_m, "Uplink\ndensity")
    panel(ax_dn, dn_d, dn_m, "Downlink\ndensity")
    ax_dn.set_xlabel("Packet size (bytes)")
    ax_dn.set_xlim(0, 1600)

    # Two annotations only; everything else is stated in the caption.
    ax_up.annotate(r"$\approx$85 B outer ACK-only",
                   xy=(95, ax_up.get_ylim()[1] * 0.55),
                   xytext=(300, ax_up.get_ylim()[1] * 0.55),
                   fontsize=6, color="darkred", va="center",
                   arrowprops=dict(arrowstyle="->", color="darkred", lw=0.8))
    # Placed mid-panel, well below the legend: at upper-right it collided with
    # the legend text on the rendered page.
    ax_dn.annotate("full MTU",
                   xy=(1225, ax_dn.get_ylim()[1] * 0.40),
                   xytext=(560, ax_dn.get_ylim()[1] * 0.40),
                   fontsize=6, color="navy", va="center",
                   arrowprops=dict(arrowstyle="->", color="navy", lw=0.8))

    fig.tight_layout(pad=0.25, h_pad=0.4)
    out = os.path.join(OUT_DIR, "short_size_distribution.png")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = load_data()
    fig_fingerprint(data)
    fig_size_distribution(data)
    print("Done.")


if __name__ == "__main__":
    main()
