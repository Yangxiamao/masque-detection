"""[2026-10-12 ICOIN r2 #1/#2] 第二个检测器: RandomForest 版的表 I 与表 II。

为什么: NetMamba+ 读字节的能力弱 (bytes-only 71.83%), 它的"下降"可能只是模型局限。
"充分"只需一个检测器做到; "必要"要看最强的检测器能否补偿 (PROJECT_LOG 2026-10-12)。
本脚本在与 NetMamba+ 完全相同的折数据、相同输入 (5x320 B + 20 signed size + 20 interval,
见 rf_on_icoin_r2_folds.feats) 上训练 RF(300), 逐样本预测落盘, 做配对 McNemar。

  table1: 每个条件 = 用该条件折的 train+valid 训练, 在该条件折的 test 上评估
          (与 NetMamba+ 的消融协议相同: 模型从头看不到被遮蔽的通道)
  table2: 检测器 clean (= full 折) / advpad / advshape / advmo 的 train+valid 训练,
          在 e6(_r)_adv_evals/<variant> 的 test 上评估 (与 run_e6_adv_kfold.py 相同)

    python experiments/rf_attack_suite.py --labels orig --part table1
    python experiments/rf_attack_suite.py --labels relabel --part table2
输出 outputs/icoin_r2/<labels>/rf_table{1,2}.json (+ rf_preds/<run>.json)
"""

import argparse
import json
import math
import os
import sys

import numpy as np
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestClassifier

_ROOT = os.environ.get("MASQUE_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "experiments"))
from rf_on_icoin_r2_folds import feats  # noqa: E402

PROC = os.path.join(_ROOT, "data", "processed")
E3 = {"orig": "e3_folds", "relabel": "e3r_folds"}
ADV_FOLDS = {"orig": "e6_adv_folds", "relabel": "e6r_adv_folds"}
ADV_EVALS = {"orig": "e6_adv_evals", "relabel": "e6r_adv_evals"}

# 表 I: (行名, 折根相对 PROC 的路径模板)
TABLE1 = [
    ("full",              "{e3}/full"),
    ("no_intervals",      "{e3}/no_intervals"),
    ("no_direction",      "{e3}/no_direction"),
    ("no_bytes",          "{e3}/no_bytes"),
    ("no_sizes",          "{e3}/no_sizes"),
    ("nolen_pad_nobytes", "icoin_r2_folds/{labels}/nolen_pad_nobytes"),
    ("suff_mag",          "{e3}/suff_mag"),
    ("suff_dir",          "{e3}/suff_dir"),
    ("nosizes_nobytes",   "{e3}/nosizes_nobytes"),
    ("suff_bytes",        "{e3}/suff_bytes"),
]
NECESSITY = ["no_intervals", "no_direction", "no_bytes", "no_sizes", "nolen_pad_nobytes"]
VARIANTS = ["t1_clean", "t_shape", "t2_pad", "t2_strict", "t3_padshape", "t3_absdir", "t3_noint", "t2_pad_mo"]


def key(it):
    return os.path.basename(it["pcap_file"].replace("\\", "/"))


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fit(train_items):
    X = np.array([feats(i) for i in train_items]); y = np.array([i["label"] for i in train_items])
    return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0).fit(X, y)


def predict(clf, items):
    pr = clf.predict(np.array([feats(i) for i in items]))
    return {key(i): {"pred": int(p), "label": int(i["label"])} for i, p in zip(items, pr)}


def wilson(k, n, z=1.959963984540054):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [(c - h) / d, (c + h) / d]


def summ(oof):
    cm = [[0, 0], [0, 0]]
    for v in oof.values():
        cm[v["label"]][v["pred"]] += 1
    n = len(oof); ok = cm[0][0] + cm[1][1]
    return {"n": n, "acc": ok / n, "ci95": wilson(ok, n), "cm": cm}


def mcnemar(a, b):
    assert set(a) == set(b)
    bb = sum(1 for k in a if a[k]["pred"] == a[k]["label"] and b[k]["pred"] != b[k]["label"])
    cc = sum(1 for k in a if a[k]["pred"] != a[k]["label"] and b[k]["pred"] == b[k]["label"])
    return {"b_only_A": bb, "c_only_B": cc, "p": binomtest(bb, bb + cc, 0.5).pvalue if bb + cc else 1.0}


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); m = len(ps); adj = [0.0] * m; prev = 0.0
    for r, i in enumerate(order):
        prev = max(prev, min(1.0, (m - r) * ps[i])); adj[i] = prev
    return adj


def table1(labels, out_dir):
    oofs, res = {}, {}
    for name, tmpl in TABLE1:
        root = os.path.join(PROC, tmpl.format(e3=E3[labels], labels=labels))
        if not os.path.isdir(root):
            print(f"  (缺 {root}, 跳过)"); continue
        oof = {}
        for k in range(5):
            d = os.path.join(root, f"fold{k}")
            clf = fit(load(os.path.join(d, "data-train.json")) + load(os.path.join(d, "data-valid.json")))
            oof.update(predict(clf, load(os.path.join(d, "data-test.json"))))
        assert len(oof) == 504, (name, len(oof))
        oofs[name] = oof; res[name] = summ(oof)
        print(f"  {name:<18} {res[name]['acc']*100:6.2f}%  cm={res[name]['cm']}", flush=True)
    tests = []
    for c in NECESSITY:
        if c in oofs and "full" in oofs:
            r = mcnemar(oofs["full"], oofs[c]); r.update(A="full", B=c); tests.append(r)
    for r, a in zip(tests, holm([t["p"] for t in tests])):
        r["p_holm"] = a
    return res, tests, oofs


def table2(labels, out_dir):
    dets = {"clean": os.path.join(PROC, E3[labels], "full")}
    for d in ("advpad", "advshape", "advmo"):
        dets[d] = os.path.join(PROC, ADV_FOLDS[labels], d)
    res, oofs = {}, {}
    for det, droot in dets.items():
        if not os.path.isdir(droot):
            print(f"  (缺 {droot}, 跳过)"); continue
        per_var = {v: {} for v in VARIANTS}
        for k in range(5):
            d = os.path.join(droot, f"fold{k}")
            clf = fit(load(os.path.join(d, "data-train.json")) + load(os.path.join(d, "data-valid.json")))
            for v in VARIANTS:
                ev = os.path.join(PROC, ADV_EVALS[labels], v, f"fold{k}", "data-test.json")
                if os.path.exists(ev):
                    per_var[v].update(predict(clf, load(ev)))
        for v, oof in per_var.items():
            if len(oof) == 504:
                res[f"{det}|{v}"] = summ(oof); oofs[f"{det}|{v}"] = oof
        print(f"  {det:<9} " + "  ".join(f"{v}={res[f'{det}|{v}']['acc']*100:.2f}"
                                          for v in VARIANTS if f"{det}|{v}" in res), flush=True)
    return res, oofs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, choices=list(E3))
    ap.add_argument("--part", required=True, choices=["table1", "table2"])
    args = ap.parse_args()
    out_dir = os.path.join(_ROOT, "outputs", "icoin_r2", args.labels)
    os.makedirs(os.path.join(out_dir, "rf_preds"), exist_ok=True)
    n_masque = None
    if args.part == "table1":
        res, tests, oofs = table1(args.labels, out_dir)
        out = {"conditions": res, "mcnemar_vs_full_holm_over_necessity_rows": tests}
        for t in tests:
            print(f"  full vs {t['B']:<18} b={t['b_only_A']:<3} c={t['c_only_B']:<3} p={t['p']:.3g} p_holm={t['p_holm']:.3g}")
    else:
        res, oofs = table2(args.labels, out_dir)
        out = {"table": res}
    for name, oof in oofs.items():
        with open(os.path.join(out_dir, "rf_preds", f"{args.part}_{name.replace('|', '__')}.json"), "w") as f:
            json.dump(oof, f)
        n_masque = sum(v["label"] for v in oof.values())
    out.update(labels=args.labels, majority=(max(n_masque, 504 - n_masque) / 504 if n_masque is not None else None),
               detector="RandomForest(n_estimators=300, random_state=0), features = rf_on_icoin_r2_folds.feats")
    p = os.path.join(out_dir, f"rf_{args.part}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("->", p)


if __name__ == "__main__":
    main()
