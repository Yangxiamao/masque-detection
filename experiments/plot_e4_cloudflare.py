"""[2026-08-23] E4 结果图: 留出 Direct->Cloudflare + leave-one-group-out。

图表约定 (CLAUDE.md 规则 4): 输出 docs/04_Meetings_and_Weekly_Notes/plots/,
Direct = cornflowerblue, MASQUE = tomato, dpi=150, **全英文标注** (WSL 无中日文字体)。

    wsl -e bash -lic 'conda activate MAMBA; cd /mnt/d/zemi/2026/dataset; \
        python experiments/plot_e4_cloudflare.py'
"""

import json
import os

import matplotlib
matplotlib.use("Agg")   # 无显示环境
import matplotlib.pyplot as plt

_ROOT = os.environ.get("MASQUE_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
SUMMARY = os.path.join(_ROOT, "outputs", "e4_cloudflare", "summary_e4.json")
PLOT_DIR = os.path.join(_ROOT, "docs", "04_Meetings_and_Weekly_Notes", "plots")

BLUE = "cornflowerblue"
RED = "tomato"
GREY = "0.6"


def err(recall, ci):
    """把 Wilson 区间转成 errorbar 需要的 (下, 上) 半长, 并夹到非负。"""
    return [[max(0.0, (recall - ci[0]) * 100)], [max(0.0, (ci[1] - recall) * 100)]]


def main():
    with open(SUMMARY, encoding="utf-8") as f:
        s = json.load(f)
    res = s["conditions"]
    ref = s.get("e3_full_reference")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8),
                                   gridspec_kw={"width_ratios": [1, 1.55]})

    # ---------------- 左: E4-A ----------------
    order = [("cf_holdout", "Held-out set =\nDirect flows to\nCloudflare (n=%d)"),
             ("nocf_holdout", "Control: held-out set =\nDirect flows to\nnon-Cloudflare (n=%d)")]
    xs, labels, vals, errs, cols = [], [], [], [[], []], []
    for i, (k, lab) in enumerate(order):
        if k not in res:
            continue
        r = res[k]
        xs.append(i)
        labels.append(lab % r["direct_n"])
        vals.append(r["direct_recall"] * 100)
        e = err(r["direct_recall"], r["direct_recall_ci95"])
        errs[0].append(e[0][0]); errs[1].append(e[1][0])
        cols.append(RED if k == "cf_holdout" else BLUE)

    ax1.bar(xs, vals, yerr=errs, color=cols, width=0.55, capsize=6,
            edgecolor="black", linewidth=0.6, error_kw={"ecolor": "0.25", "lw": 1.2})
    for x, v, e_hi, r in zip(xs, vals, errs[1], [res[k] for k, _ in order if k in res]):
        ax1.text(x, v + e_hi + 2.0,
                 "%.1f%%\n(%d/%d)" % (v, r["direct_correct"], r["direct_n"]),
                 ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    if ref:
        # 参照線のラベルは軸内に置くと棒に被るので凡例へ (軸の外側下)
        ax1.axhline(ref["direct_recall"] * 100, color=GREY, ls="--", lw=1.4,
                    label="E3 5-fold OOF, standard split (%.1f%%)" % (ref["direct_recall"] * 100))
        ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24),
                   fontsize=8.5, framealpha=1.0)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("Direct recall (%)  —  held-out Direct flows\nstill classified as Direct")
    ax1.set_ylim(0, 122)
    ax1.set_yticks(range(0, 101, 20))
    ax1.set_title("(a) E4-A  Cloudflare-destined Direct flows held out\n"
                  "(zero such flows in training)", fontsize=10.5)
    ax1.grid(axis="y", ls=":", alpha=0.5)
    ax1.set_axisbelow(True)

    # ---------------- 右: E4-B ----------------
    go = [(k, v) for k, v in res.items() if v["mode"].startswith("groupout:")]
    go.sort(key=lambda x: -x[1]["direct_n"])
    cf_groups = {"104.16", "104.17", "104.18", "104.19", "172.64", "170.114"}

    gx, glab, gval, gerr, gcol = [], [], [], [[], []], []
    for i, (_k, r) in enumerate(go):
        g = r["mode"].split(":", 1)[1]
        gx.append(i)
        glab.append("%s%s\n(n=%d)" % (g, " CF" if g in cf_groups else "", r["direct_n"]))
        gval.append(r["direct_recall"] * 100)
        e = err(r["direct_recall"], r["direct_recall_ci95"])
        gerr[0].append(e[0][0]); gerr[1].append(e[1][0])
        gcol.append(RED if g in cf_groups else BLUE)

    ax2.bar(gx, gval, yerr=gerr, color=gcol, width=0.62, capsize=5,
            edgecolor="black", linewidth=0.6, error_kw={"ecolor": "0.25", "lw": 1.1})
    for x, v, e_hi, (_k, r) in zip(gx, gval, gerr[1], go):
        ax2.text(x, v + e_hi + 2.0, "%d/%d" % (r["direct_correct"], r["direct_n"]),
                 ha="center", va="bottom", fontsize=9)

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [Patch(facecolor=RED, edgecolor="black", label="Cloudflare-owned"),
               Patch(facecolor=BLUE, edgecolor="black", label="other operators")]
    pool = s.get("groupout_pooled")
    if pool:
        ax2.axhline(pool["pooled_recall"] * 100, color="0.35", ls="-", lw=1.4)
        handles.append(Line2D([0], [0], color="0.35", lw=1.4,
                              label="pooled over all groups (%.1f%%, %d/%d)"
                                    % (pool["pooled_recall"] * 100,
                                       pool["pooled_correct"], pool["pooled_n"])))
    ax2.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.24),
               ncol=3, fontsize=8.5, framealpha=1.0)
    ax2.set_xticks(gx)
    ax2.set_xticklabels(glab, fontsize=8.5)
    ax2.set_ylim(0, 122)
    ax2.set_yticks(range(0, 101, 20))
    ax2.set_xlabel("Held-out destination /16 group", labelpad=6)
    ax2.set_title("(b) E4-B  Leave-one-destination-group-out\n"
                  "(asymmetric: grouping on the Direct side only)", fontsize=10.5)
    ax2.grid(axis="y", ls=":", alpha=0.5)
    ax2.set_axisbelow(True)

    fig.suptitle("E4: is the model recognising the tunnel, or just the peer being Cloudflare?",
                 fontsize=12.5, y=0.995)
    fig.tight_layout(rect=[0, 0.04, 1, 0.955])
    fig.text(0.005, 0.006,
             "Error bars: Wilson 95% CI.  Peer IPs are zeroed by the debiasing step, "
             "so the model cannot read the address itself.  n=504 flows, single capture 2026-06-16.",
             fontsize=7.5, color="0.35")

    os.makedirs(PLOT_DIR, exist_ok=True)
    p = os.path.join(PLOT_DIR, "e4_cloudflare_holdout.png")
    fig.savefig(p, dpi=150)
    print("wrote %s" % p)


if __name__ == "__main__":
    main()
