"""[2026-10-12 ICOIN r2 §7.5] 改标版: 原选模规则 (acc_first) -> 新规则 (acc_loss) 的对照汇总。

见 docs/01_Planning/Handoff_Relabel_SelRule_Rerun_2026-10-12.md §7。只读 preds / summary, CPU 几秒。

  (1) 论文每个依赖 NetMamba+ 的改标数字: 旧规则 -> 新规则, Wilson 95% CI, 混淆矩阵, 来源目录
  (2) p 值只在同一规则的结果之间计算 (不跨规则):
        表 I Holm-11 族 (analyze_e3_mcnemar 的 10 个条件 + No-Length, 各 vs Full)
        IV-C 抖动: 未校正 exact McNemar (jit* vs Timing only, jit1ms_full vs Full)
        表 II: runner 的 24 族 + 论文的 21 族 (= 不含 t3_noint 的 7 个测试变体 x 3 个对抗检测器)
        #1 / #2 / C 系列: run_icoin_r2.py 汇总里原有的族 (直接摘录)
  (3) 一致性: 样本集合 504 / 标签 Direct 245, MASQUE 259; 每个条件的 preds_acc_first.json
      与旧目录的 preds 逐条比较 (= 旧规则复现率); 各折选中的 epoch

    python experiments/summarize_relabel_sel.py   # -> outputs/relabel_sel/summary_relabel_sel.json
"""

import json
import os
import sys
import time

_ROOT = os.environ.get("MASQUE_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "experiments"))
from run_kfold_e3 import holm, mcnemar, wilson_ci  # noqa: E402

O = os.path.join(_ROOT, "outputs")
DIRS = {   # 旧规则 (acc_first) / 新规则 (acc_loss)
    "e3": (os.path.join(O, "e3r_kfold"), os.path.join(O, "e3r_kfold_sel")),
    "ic": (os.path.join(O, "icoin_r2", "relabel"), os.path.join(O, "icoin_r2", "relabel_sel")),
    "e6": (os.path.join(O, "e6r_adv_kfold"), os.path.join(O, "e6r_adv_kfold_sel")),
    "e4": (os.path.join(O, "e4_cloudflare_relabel"), os.path.join(O, "e4_cloudflare_relabel_sel")),
    "op": (os.path.join(O, "e6_operating_points_relabel"), os.path.join(O, "e6_operating_points_relabel_sel")),
}
OUT = os.path.join(O, "relabel_sel")
N_SPLITS = 5

ICOIN = ["t3dir_raw", "t3dir_san", "t3dir_strict", "nolen_pad", "nolen_pad_nobytes", "nolen_pad_san",
         "t3dir_c123_san", "t3dir_c123_hdr", "t3dir_c123_hdrtype"]
TABLE1 = [("full", "Full"), ("no_intervals", "No-Intervals"), ("no_direction", "No-Direction"),
          ("no_bytes", "No-Bytes"), ("no_sizes", "No-Sizes"), ("nolen_pad_nobytes", "No-Length"),
          ("suff_mag", "Length only"), ("suff_dir", "Direction only"),
          ("nosizes_nobytes", "Timing only"), ("suff_bytes", "Bytes only")]
CONST = [("c1", "C1 MTU"), ("c2", "C2 CID"), ("c3", "C3 Retry"), ("c123", "C123"),
         ("c123_nobytes", "C123 + No-Bytes")]
FAMILY11 = ["no_intervals", "no_direction", "no_bytes", "no_sizes", "nosizes_nobytes", "nolen_pad_nobytes",
            "c1", "c2", "c3", "c123", "c123_nobytes"]
JITTER = [("jit50us", "nosizes_nobytes"), ("jit200us", "nosizes_nobytes"), ("jit1ms", "nosizes_nobytes"),
          ("jit5ms", "nosizes_nobytes"), ("jit1ms_full", "full")]
E3_CONDS = [c for c, _ in TABLE1 if c not in ICOIN] + [c for c, _ in CONST] + [c for c, _ in JITTER]
E6_DETS = ["advpad", "advshape", "advmo"]
E6_PAPER_EXCLUDE = {"t3_noint"}   # 论文表 II 的 21 族 = 草稿版表 II 的 7 列 x 3 检测器


def rel(p):
    return os.path.relpath(p, _ROOT).replace("\\", "/")


def cond_dir(cond, new):
    return os.path.join(DIRS["ic" if cond in ICOIN else "e3"][1 if new else 0], cond)


def load_folds(d, name="preds.json"):
    oof = {}
    for k in range(N_SPLITS):
        with open(os.path.join(d, f"fold{k}", name), encoding="utf-8") as f:
            part = json.load(f)
        assert not (set(part) & set(oof)), f"{d} fold{k}: 重复样本"
        oof.update(part)
    return oof


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def stats(oof):
    n = len(oof)
    ok = sum(v["pred"] == v["label"] for v in oof.values())
    cm = [[0, 0], [0, 0]]
    for v in oof.values():
        cm[v["label"]][v["pred"]] += 1
    lo, hi = wilson_ci(ok, n)
    return {"n": n, "correct": ok, "acc": ok / n, "ci95": [lo, hi], "cm": cm}


def repro(first, old):
    """新规则那次训练里原规则 ckpt 的预测 vs 旧目录的预测 (训练可复现性)。"""
    assert set(first) == set(old), "样本集合不同"
    return {"n": len(old),
            "pred_mismatch": sum(first[x]["pred"] != old[x]["pred"] for x in old),
            "label_mismatch": sum(first[x]["label"] != old[x]["label"] for x in old),
            "max_abs_prob1_diff": max(abs(first[x]["prob1"] - old[x]["prob1"]) for x in old)}


def sel_epochs(run_dirs):
    out = {"acc_first": [], "acc_loss": []}
    for d in run_dirs:
        s = load_json(os.path.join(d, "selrule.json"))["selected_epoch"]
        out["acc_first"].append(s["acc_first"])
        out["acc_loss"].append(s["acc_loss"])
    return out


def family(oofs, base, conds):
    tests = []
    for c in conds:
        t = mcnemar(oofs[base], oofs[c])
        t.update({"A": base, "B": c})
        tests.append(t)
    for t, adj in zip(tests, holm([t["p"] for t in tests])):
        t["p_holm"] = adj
    return tests


def fmt(s):
    return f"{s['acc']*100:6.2f} [{s['ci95'][0]*100:.2f}, {s['ci95'][1]*100:.2f}] {s['cm']}"


def main():
    res = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
           "rules": {"old": "acc_first: valid acc 第一次达到最高 (fine-tune.py 原规则)",
                     "new": "acc_loss: valid acc 最高, 打平取 valid loss 最低 (预先指定)"},
           "dirs": {k: {"old": rel(a), "new": rel(b)} for k, (a, b) in DIRS.items()}}

    # ---------------- 逐条件 OOF: 新 / 旧 / 新训练里的 acc_first ----------------
    conds = list(dict.fromkeys([c for c, _ in TABLE1] + E3_CONDS + ICOIN))
    new, old, cons = {}, {}, {}
    for c in conds:
        new[c] = load_folds(cond_dir(c, True))
        old[c] = load_folds(cond_dir(c, False))
        first = load_folds(cond_dir(c, True), "preds_acc_first.json")
        n_direct = sum(v["label"] == 0 for v in new[c].values())
        cons[c] = {"n": len(new[c]), "direct": n_direct, "masque": len(new[c]) - n_direct,
                   "labels_equal_old": all(new[c][x]["label"] == old[c][x]["label"] for x in old[c]),
                   "repro_acc_first_vs_old": repro(first, old[c]),
                   "selected_epochs": sel_epochs([os.path.join(cond_dir(c, True), f"fold{k}")
                                                  for k in range(N_SPLITS)]),
                   "new_dir": rel(cond_dir(c, True)), "old_dir": rel(cond_dir(c, False))}
        assert len(new[c]) == 504 and set(new[c]) == set(new["full"]), c
        assert n_direct == 245 and cons[c]["labels_equal_old"], c

    res["per_condition"] = {c: {"old": stats(old[c]), "new": stats(new[c]), **cons[c]} for c in conds}

    # ---------------- 表 I Holm-11 族 (每套规则各算各的) ----------------
    res["table1_family11"] = {"old": family(old, "full", FAMILY11), "new": family(new, "full", FAMILY11),
                              "note": "exact McNemar vs Full, Holm over 11 (6 in Table I + 5 in VI-C)"}

    # ---------------- IV-C 抖动: 未校正 exact McNemar ----------------
    res["jitter"] = {}
    for rule, oofs in (("old", old), ("new", new)):
        res["jitter"][rule] = []
        for c, base in JITTER:
            t = mcnemar(oofs[base], oofs[c])
            t.update({"A": base, "B": c})
            res["jitter"][rule].append(t)

    # ---------------- #1 / #2 / C 系列: runner 汇总摘录 ----------------
    res["icoin_r2_families"] = {r: load_json(os.path.join(DIRS["ic"][i], "summary.json"))["mcnemar_families"]
                                for i, r in enumerate(("old", "new"))}

    # ---------------- 表 II ----------------
    e6 = {}
    for i, rule in enumerate(("old", "new")):
        s = load_json(os.path.join(DIRS["e6"][i], "summary_e6_adv_kfold.json"))
        variants = sorted({v["variant"] for v in s["table"].values()})
        oofs = {}
        for det in ["clean"] + E6_DETS:
            for var in variants:
                oofs[(det, var)] = {}   # 表 II 的 preds 按 preds/<det>/<var>/fold{k}.json 存放
                for k in range(N_SPLITS):
                    oofs[(det, var)].update(load_json(os.path.join(DIRS["e6"][i], "preds", det, var, f"fold{k}.json")))
        fam21 = []
        for var in variants:
            if var in E6_PAPER_EXCLUDE:
                continue
            for det in E6_DETS:
                t = mcnemar(oofs[("clean", var)], oofs[(det, var)])
                t.update({"variant": var, "A": "clean", "B": det})
                fam21.append(t)
        for t, adj in zip(fam21, holm([t["p"] for t in fam21])):
            t["p_holm21"] = adj
        e6[rule] = {"table": {k: {kk: v[kk] for kk in ("acc", "ci95", "cm", "correct", "n")}
                              for k, v in s["table"].items()},
                    "runner_family24": s["mcnemar_clean_vs_adv"],
                    "paper_family21": fam21,
                    "transfer_advpad_vs_advshape_on_t3": s["transfer_advpad_vs_advshape_on_t3"]}
    # 对抗检测器: 新训练里原规则 ckpt 在干净 test 上的预测 vs 旧 preds/<det>/t1_clean
    e6["consistency"] = {}
    for det in E6_DETS:
        first, prev = {}, {}
        for k in range(N_SPLITS):
            first.update(load_json(os.path.join(DIRS["e6"][1], det, f"fold{k}", "preds_acc_first.json")))
            prev.update(load_json(os.path.join(DIRS["e6"][0], "preds", det, "t1_clean", f"fold{k}.json")))
        e6["consistency"][det] = {
            "repro_acc_first_vs_old": repro(first, prev),
            "selected_epochs": sel_epochs([os.path.join(DIRS["e6"][1], det, f"fold{k}") for k in range(N_SPLITS)])}
    res["table2"] = e6

    # ---------------- E4 ----------------
    e4 = {}
    for i, rule in enumerate(("old", "new")):
        s = load_json(os.path.join(DIRS["e4"][i], "summary_e4.json"))
        e4[rule] = {"conditions": {k: {kk: v[kk] for kk in ("acc", "acc_ci95", "cm", "direct_n", "direct_correct",
                                                            "direct_recall", "direct_recall_ci95",
                                                            "masque_n", "masque_correct")}
                                   for k, v in s["conditions"].items()},
                    "groupout_pooled": s["groupout_pooled"], "tests": s["tests"]}
    e4["consistency"] = {}
    for name in e4["new"]["conditions"]:
        first = load_json(os.path.join(DIRS["e4"][1], name, "preds_acc_first.json"))
        prev = load_json(os.path.join(DIRS["e4"][0], name, "preds.json"))
        sel = load_json(os.path.join(DIRS["e4"][1], name, "selrule.json"))["selected_epoch"]
        e4["consistency"][name] = {"repro_acc_first_vs_old": repro(first, prev),
                                   "selected_epoch": {"acc_first": sel["acc_first"], "acc_loss": sel["acc_loss"]}}
    res["e4"] = e4

    # ---------------- 工作点 ----------------
    res["operating_points_full"] = {r: load_json(os.path.join(DIRS["op"][i], "operating_points_full.json"))
                                    for i, r in enumerate(("old", "new"))}

    # ---------------- 复现率汇总 ----------------
    rp = [(c, cons[c]["repro_acc_first_vs_old"]) for c in conds] + \
         [(f"e6:{d}", e6["consistency"][d]["repro_acc_first_vs_old"]) for d in E6_DETS] + \
         [(f"e4:{n}", e4["consistency"][n]["repro_acc_first_vs_old"]) for n in e4["consistency"]]
    res["repro_overall"] = {
        "n_runs": len(rp), "runs_with_zero_pred_mismatch": sum(r["pred_mismatch"] == 0 for _, r in rp),
        "total_pred_mismatch": sum(r["pred_mismatch"] for _, r in rp),
        "total_predictions": sum(r["n"] for _, r in rp),
        "max_abs_prob1_diff": max(r["max_abs_prob1_diff"] for _, r in rp),
        "runs_with_mismatch": {c: r for c, r in rp if r["pred_mismatch"]}}

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "summary_relabel_sel.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    # ---------------- 打印 ----------------
    f11 = {r: {t["B"]: t for t in res["table1_family11"][r]} for r in ("old", "new")}
    print("== 表 I + VI-C (改标, n=504)  OOF [95% CI] cm | vs Full b/c p_holm(11)")
    for c, name in TABLE1 + CONST:
        o, n = res["per_condition"][c]["old"], res["per_condition"][c]["new"]
        line = f"{name:<16} old {fmt(o):<44} new {fmt(n):<44}"
        if c in f11["new"]:
            to, tn = f11["old"][c], f11["new"][c]
            line += (f" old {to['b_only_A_correct']}/{to['c_only_B_correct']} {to['p_holm']:.2g}"
                     f" | new {tn['b_only_A_correct']}/{tn['c_only_B_correct']} {tn['p_holm']:.2g}")
        print(line)
    print("\n== IV-C 抖动 (未校正 exact McNemar)")
    for r in ("old", "new"):
        for t in res["jitter"][r]:
            print(f"  {r} {t['B']:<12} vs {t['A']:<16} {res['per_condition'][t['B']][r]['acc']*100:6.2f}"
                  f"  b/c {t['b_only_A_correct']}/{t['c_only_B_correct']}  p={t['p']:.2g}")
    print("\n== #1/#2/C 系列")
    for c in ICOIN:
        print(f"  {c:<20} old {fmt(res['per_condition'][c]['old']):<44} new {fmt(res['per_condition'][c]['new'])}")
    for fam, tests in res["icoin_r2_families"]["new"].items():
        for t in tests:
            print(f"    new [{fam}] {t['A']} vs {t['B']}: {t['b_only_A_correct']}/{t['c_only_B_correct']}"
                  f" p_holm={t['p_holm_within_family']:.2g}")
    print("\n== 表 II (acc %, old -> new)")
    vars_ = sorted({k.split("|", 1)[1] for k in e6["new"]["table"]})
    for det in ["clean"] + E6_DETS:
        print(f"  {det:<9}" + "  ".join(f"{v}:{e6['old']['table'][f'{det}|{v}']['acc']*100:.2f}->"
                                        f"{e6['new']['table'][f'{det}|{v}']['acc']*100:.2f}" for v in vars_))
    print("\n== E4")
    for r in ("old", "new"):
        cf = e4[r]["conditions"]["cf_holdout"]
        gp = e4[r]["groupout_pooled"]
        print(f"  {r}: cf_holdout Direct {cf['direct_correct']}/{cf['direct_n']}; groupout pooled "
              f"{gp['pooled_correct']}/{gp['pooled_n']} = {gp['pooled_recall']*100:.1f}% "
              f"[{gp['pooled_ci95'][0]*100:.1f}, {gp['pooled_ci95'][1]*100:.1f}]")
    print("\n== 工作点")
    for r in ("old", "new"):
        op = res["operating_points_full"][r]
        print(f"  {r}: AUC {op['auc']}  acc {op['accuracy']}  0.5 点 {op['operating_point_0.5']}")
    ro = res["repro_overall"]
    print(f"\n== 旧规则复现: {ro['runs_with_zero_pred_mismatch']}/{ro['n_runs']} 个运行逐条一致; "
          f"不一致 {ro['total_pred_mismatch']}/{ro['total_predictions']} 条预测; "
          f"prob1 最大差 {ro['max_abs_prob1_diff']:.1e}")
    for c, r in ro["runs_with_mismatch"].items():
        print(f"   ! {c}: {r}")
    print(f"\n已写入 {path}")


if __name__ == "__main__":
    main()
