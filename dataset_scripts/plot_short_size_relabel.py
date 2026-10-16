#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
[2026-10-13 ICOIN r2 #3] Fig. 3 (packet-size distribution) under the corrected
labels: the 11 flows of outputs/e3_kfold/error_analysis_e3.json
label_noise_flows (peer outside WARP 162.159) become Direct.  Reuses
plot_short_paper_figs.fig_size_distribution unchanged; writes
short_size_distribution_relabel.png (the original-label figure is kept).

Also prints the window statistics quoted in paper_conf Sec. III-C for both
label sets, so the original-label numbers double as a reproduction check.

Run in WSL conda MAMBA env:
  wsl -e bash -lic 'conda activate MAMBA; cd /mnt/d/zemi/2026/dataset/dataset_scripts; python plot_short_size_relabel.py'
"""
import copy
import json
import os
import shutil
import tempfile

import numpy as np

import plot_short_paper_figs as base

ERR = os.path.join(base._ROOT, "outputs", "e3_kfold", "error_analysis_e3.json")
OUT = os.path.join(base.OUT_DIR, "short_size_distribution_relabel.png")


def relabel(data):
    with open(ERR, encoding="utf-8") as f:
        noise = set(json.load(f)["label_noise_flows"])
    assert len(noise) == 11
    out = copy.deepcopy(data)
    n = 0
    for r in out:
        if os.path.basename(r["pcap_file"].replace("\\", "/")) in noise:
            assert r["label"] == 1
            r["label"] = 0
            n += 1
    assert n == 11, n
    return out


def stats(data, tag):
    up_d, up_m, dn_d, dn_m = [], [], [], []
    for rec in data:
        for v in base.parse_signed(rec):
            if v > 0:
                (up_d if rec["label"] == 0 else up_m).append(v)
            elif v < 0:
                (dn_d if rec["label"] == 0 else dn_m).append(-v)
    dn_d, dn_m, up_m = map(np.array, (dn_d, dn_m, up_m))
    n0 = sum(r["label"] == 0 for r in data)
    res = {
        "flows_direct": int(n0), "flows_masque": int(len(data) - n0),
        "n_up_direct": len(up_d), "n_up_masque": int(len(up_m)),
        "n_dn_direct": int(len(dn_d)), "n_dn_masque": int(len(dn_m)),
        "direct_dn_frac_1240_1300": float(np.mean((dn_d >= 1240) & (dn_d <= 1300))),
        # PROJECT_LOG 2026-09-14: the paper's "69%" is this (>= 1200 B), not
        # the 1240-1300 B band (55.2%).
        "direct_dn_frac_ge_1200": float(np.mean(dn_d >= 1200)),
        "direct_dn_median": float(np.median(dn_d)),
        "masque_dn_frac_below_120": float(np.mean(dn_m < 120)),
        "masque_dn_median": float(np.median(dn_m)),
        "masque_up_frac_80_90": float(np.mean((up_m >= 80) & (up_m <= 90))),
    }
    print(tag, json.dumps(res, indent=1))
    return res


def main():
    data = base.load_data()
    res = {"orig": stats(data, "orig"), "relabel": stats(relabel(data), "relabel")}
    tmp = tempfile.mkdtemp()
    base.OUT_DIR = tmp
    base.fig_size_distribution(relabel(data))
    shutil.copy(os.path.join(tmp, "short_size_distribution.png"), OUT)
    print("Saved:", OUT)
    js = OUT.replace(".png", "_stats.json")
    with open(js, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print("Saved:", js)


if __name__ == "__main__":
    main()
