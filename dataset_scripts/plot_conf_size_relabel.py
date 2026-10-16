#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
[2026-10-13 ICOIN format pass] Fig. 3 of paper_conf (packet-size distribution,
corrected labels) redrawn at its FINAL printed size: one IEEE column wide
(3.5 in, included with width=\\columnwidth) and 1.67 in tall, i.e. the same
column height the old figure took (3.35 x 2.05 in shown at 0.78 column width).
At that old scale its 6-7 pt type printed at ~5-6 pt; here every label is
8 pt (IEEE recommends 8-10 pt) and serif, to match the Times body text.

Same data, relabelling and histogram as plot_short_size_relabel.py /
plot_short_paper_figs.fig_size_distribution (bins 0-1600 B, 99 bins, density);
only the size, fonts and annotation geometry differ.  The window statistics
quoted in the text are unaffected (they come from
short_size_distribution_relabel_stats.json).

Run in WSL conda MAMBA env:
  wsl -e bash -lic 'conda activate MAMBA; cd /mnt/d/zemi/2026/dataset/dataset_scripts; python plot_conf_size_relabel.py'
"""
import os

import numpy as np

import plot_short_paper_figs as base
from plot_short_size_relabel import relabel

import matplotlib.pyplot as plt  # backend already set to Agg by base

OUT = os.path.join(base.OUT_DIR, "conf_size_distribution_relabel.png")
W, H = 3.5, 1.67   # inches, final size on the page
PT = 8             # every label, IEEE minimum

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Nimbus Roman No9 L", "Times New Roman",
                   "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": PT,
    "axes.labelsize": PT,
    "xtick.labelsize": PT,
    "ytick.labelsize": PT,
    "legend.fontsize": PT,
})


def main():
    data = relabel(base.load_data())
    up_d, up_m, dn_d, dn_m = [], [], [], []
    for rec in data:
        for v in base.parse_signed(rec):
            if v > 0:
                (up_d if rec["label"] == 0 else up_m).append(v)
            elif v < 0:
                (dn_d if rec["label"] == 0 else dn_m).append(-v)
    print(f"uplink   Direct {len(up_d)}  MASQUE {len(up_m)}")
    print(f"downlink Direct {len(dn_d)}  MASQUE {len(dn_m)}")

    bins = np.linspace(0, 1600, 100)
    fig, (ax_up, ax_dn) = plt.subplots(2, 1, figsize=(W, H), sharex=True)

    def panel(ax, d, m, ylab):
        ax.hist(d, bins=bins, alpha=0.65, density=True, color=base.DIRECT_COLOR,
                label=f"Direct  n={len(d):,}", linewidth=0)
        ax.hist(m, bins=bins, alpha=0.65, density=True, color=base.MASQUE_COLOR,
                label=f"MASQUE  n={len(m):,}", linewidth=0)
        ax.set_ylabel(ylab, labelpad=2)
        ax.legend(loc="upper center", bbox_to_anchor=(0.52, 1.02), frameon=False,
                  handlelength=1.0, borderpad=0.1, labelspacing=0.1,
                  handletextpad=0.4)
        ax.tick_params(length=2.5, width=0.6, pad=1.5)
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.6)

    panel(ax_up, up_d, up_m, "Uplink\ndensity")
    panel(ax_dn, dn_d, dn_m, "Downlink\ndensity")
    ax_dn.set_xlabel("Packet size (bytes)", labelpad=1)
    ax_dn.set_xlim(0, 1600)

    # Arrow tips on the bars they name: the ~85 B uplink spike and the full-MTU
    # downlink cluster.  Text sits between the legend and the tall bars.
    y_up = ax_up.get_ylim()[1]
    ax_up.annotate(r"$\approx$85 B outer ACK-only", xy=(100, y_up * 0.42),
                   xytext=(330, y_up * 0.42), color="darkred", va="center",
                   arrowprops=dict(arrowstyle="->", color="darkred", lw=0.8,
                                   shrinkA=1, shrinkB=0))
    y_dn = ax_dn.get_ylim()[1]
    ax_dn.annotate("full MTU", xy=(1215, y_dn * 0.40),
                   xytext=(760, y_dn * 0.40), color="navy", va="center",
                   arrowprops=dict(arrowstyle="->", color="navy", lw=0.8,
                                   shrinkA=1, shrinkB=0))

    # right=0.965 leaves room for the "1600" tick label (clipped at 0.985).
    fig.subplots_adjust(left=0.115, right=0.965, top=0.985, bottom=0.215,
                        hspace=0.12)
    fig.savefig(OUT, dpi=600)
    plt.close(fig)
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
