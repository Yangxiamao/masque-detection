"""[2026-08-23] E3 附加分析: OOF 错误流与 11 条标签噪声流的重合度。

E3_handoff.md §9: 8/18 已查明数据里有 11 条流标着 Masque 但对端不是 162.159/16
(disconnect 窗口内抓到的直连流量), 构成 493/504 = 97.82% 的"硬天花板"。
Full 的 OOF 错误里 Masque->Direct 方向占 9/10, 数量与之接近 —— 值得核对是不是同一批流,
而不是靠印象断言。

    python experiments/analyze_e3_errors.py
"""

import json
import os
import sys

_ROOT = os.environ.get("MASQUE_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_ROOT = os.path.join(_ROOT, "outputs", "e3_kfold")
WARP_PREFIX = "162.159."


def endpoints(basename):
    """basename: class1_Masque_iter1_<date>_<time>_<ip1(4段)>_<port1>_<ip2(4段)>_<port2>.pcap"""
    segs = basename.replace(".pcap", "").split("_")
    q = segs[-10:]
    return ".".join(q[0:4]), ".".join(q[5:9])


def is_label_noise(basename, label):
    """标 Masque(label=1) 但两端都不是 WARP 入口 162.159/16。"""
    if label != 1:
        return False
    a, b = endpoints(basename)
    return not (a.startswith(WARP_PREFIX) or b.startswith(WARP_PREFIX))


def load_oof(cid, n_splits=5):
    oof = {}
    for k in range(n_splits):
        p = os.path.join(OUT_ROOT, cid, f"fold{k}", "preds.json")
        if not os.path.exists(p):
            return None
        with open(p, encoding="utf-8") as f:
            oof.update(json.load(f))
    return oof


def main():
    conds = sys.argv[1:] or ["full", "c123", "nosizes_nobytes"]

    ref = load_oof(conds[0])
    if ref is None:
        raise SystemExit(f"缺少 {conds[0]} 的 preds.json, 先跑 run_kfold_e3.py")
    noise = {k for k, v in ref.items() if is_label_noise(k, v["label"])}

    print(f"全集 {len(ref)} 条; 标 Masque 但对端不在 162.159/16 的流: {len(noise)} 条")
    print(f"(对应天花板 {(len(ref)-len(noise))}/{len(ref)} = "
          f"{(len(ref)-len(noise))/len(ref)*100:.2f}%)\n")
    for k in sorted(noise):
        a, b = endpoints(k)
        print(f"  {k}   ({a} <-> {b})")

    print(f"\n{'='*88}")
    print(f"{'条件':<20}{'错':>5}{'其中噪声流':>12}{'其中正常流':>12}"
          f"{'噪声流被判错率':>16}{'正常流被判错率':>16}")
    print("-" * 88)
    for cid in conds:
        oof = load_oof(cid)
        if oof is None:
            print(f"(跳过未完成: {cid})")
            continue
        wrong = {k for k, v in oof.items() if v["pred"] != v["label"]}
        w_noise = wrong & noise
        w_clean = wrong - noise
        n_clean = len(oof) - len(noise)
        print(f"{cid:<20}{len(wrong):>5}{len(w_noise):>12}{len(w_clean):>12}"
              f"{len(w_noise)/len(noise)*100:>15.1f}%{len(w_clean)/n_clean*100:>15.1f}%")
    print("-" * 88)
    print("噪声流被判错率若接近 100%, 说明模型的 OOF 错误主要是数据标签本身的问题, 不是模型能力;")
    print("正常流被判错率才是可解读的泛化误差。")

    # ---- 敏感性分析 (§9: 不删这 11 条, 只作为单独一组对照报告) ----
    sys.path.insert(0, os.path.join(_ROOT, "experiments"))
    from run_kfold_e3 import wilson_ci

    print(f"\n{'='*88}")
    print("敏感性分析: 排除 11 条标签噪声流后的 OOF (n=493)")
    print("⚠️ 论文主表仍应引用全 504 条的数字; 本表只用来说明误差的来源构成。")
    print(f"{'='*88}")
    print(f"{'条件':<24}{'OOF(504)':>11}{'OOF(493, 去噪声)':>20}{'95% CI (Wilson)':>22}")
    print("-" * 88)
    rows = {}
    for cid in conds:
        oof = load_oof(cid)
        if oof is None:
            continue
        n_all = len(oof)
        ok_all = sum(v["pred"] == v["label"] for v in oof.values())
        clean = {k: v for k, v in oof.items() if k not in noise}
        ok_c = sum(v["pred"] == v["label"] for v in clean.values())
        lo, hi = wilson_ci(ok_c, len(clean))
        print(f"{cid:<24}{ok_all/n_all*100:>10.2f}%{ok_c/len(clean)*100:>19.2f}%"
              f"{f'[{lo*100:.2f}, {hi*100:.2f}]':>22}")
        rows[cid] = {"n_all": n_all, "correct_all": ok_all,
                     "n_clean": len(clean), "correct_clean": ok_c,
                     "acc_clean": ok_c / len(clean), "ci95_clean": [lo, hi],
                     "wrong_noise": sorted(set(k for k, v in oof.items()
                                               if v["pred"] != v["label"]) & noise)}
    out = os.path.join(OUT_ROOT, "error_analysis_e3.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"label_noise_flows": sorted(noise), "n_noise": len(noise),
                   "conditions": rows}, f, indent=2, ensure_ascii=False)
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
