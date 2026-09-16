# -*- coding: utf-8 -*-
"""IAT 統計ベースライン：時間情報は本当に存在しないのか。

背景:
  bi_abl_no_sizes_nobytes（時間のみ）で NetMamba+ は 52.94% = 多数クラス基線に
  塌陷した。しかし loader_data.py:180 の sigmoid(log(1+x)) 符号化により、
  50µs と 215µs の差は符号化後 4.1e-5 にまで圧縮される（値域 [0.5, 1.0] の 0.008%）。
  → モデルが時間を使えなかったのは「情報が無い」からか「符号化で潰れた」からか。

本スクリプトは NetMamba を一切使わず、生の IAT から統計量を作って
古典的分類器にかける。同一の会話レベル分割を使用（data-{train,valid,test}.json）。

判定:
  test acc >> 52.94%  → 時間に情報はある。NetMamba の陰性は符号化アーティファクト。
  test acc ~= 52.94%  → このデータでは時間の可分性が実際に弱い。

実行: wsl -e bash -lic 'conda activate MAMBA; cd /mnt/d/zemi/2026/dataset; python experiments/iat_statistical_baseline.py'
"""
import json
import os

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_ROOT = os.environ.get("MASQUE_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
DATA = os.path.join(_ROOT, "data", "processed", "quic_vs_masque_bi_feat")
OUT = os.path.join(_ROOT, "outputs", "iat_statistical_baseline")
os.makedirs(OUT, exist_ok=True)

SEED = 42
# NetMamba が実際に見ている長さ。これに揃えた条件が本命の対照。
SEQ_LEN = 20
FLOOR = 1e-6  # log を取る前の下限（1µs）

# 対数十年ビン（7/14 の IAT 分布図の構造をそのまま特徴にする）
DECADE_EDGES = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]


def load_split(name):
    with open(os.path.join(DATA, f"data-{name}.json")) as f:
        return json.load(f)


def iats_of(item, limit):
    """先頭パケットの 0 を除いた IAT 列。limit=None なら全部。"""
    vals = [float(v) for v in item["intervals"].split()]
    if limit is not None:
        vals = vals[:limit]
    return np.asarray(vals[1:], dtype=np.float64)


def featurize(item, limit):
    x = iats_of(item, limit)
    if x.size == 0:
        return np.zeros(20, dtype=np.float64)

    lg = np.log10(np.maximum(x, FLOOR))
    qs = np.percentile(lg, [10, 25, 50, 75, 90])
    hist = np.histogram(x, bins=[0.0] + DECADE_EDGES + [np.inf])[0] / x.size

    return np.concatenate([
        [lg.mean(), lg.std(), lg.min(), lg.max(), qs[3] - qs[1]],
        qs,
        hist,                                   # 7 次元
        [x.sum(), x.size / max(x.sum(), 1e-9)],  # 継続時間・パケットレート
    ])


FEAT_NAMES = (
    ["log_mean", "log_std", "log_min", "log_max", "log_iqr"]
    + [f"log_q{q}" for q in (10, 25, 50, 75, 90)]
    + ["frac_<10us", "frac_10-100us", "frac_100us-1ms", "frac_1-10ms",
       "frac_10-100ms", "frac_0.1-1s", "frac_>1s"]
    + ["duration", "pkt_rate"]
)
# 特徴グループ（どの時間情報が効いているかの帰属用）
GROUPS = {
    "分布の形のみ (log 統計 + 十年ビン)": list(range(0, 17)),
    "log 統計のみ": list(range(0, 10)),
    "十年ビンのみ": list(range(10, 17)),
    "継続時間・レートのみ": [17, 18],
}


def build(split_items, limit):
    X = np.vstack([featurize(it, limit) for it in split_items])
    y = np.asarray([it["label"] for it in split_items])
    return X, y


def attribution(tr, va, te, limit):
    """どの時間特徴が効いているか。サイト交絡（RTT）の疑いを切り分けるため。"""
    print(f"\n{'=' * 68}\n■ 帰属分析（first20・RandomForest）\n{'=' * 68}")
    Xfit, yfit = build(tr + va, limit)
    Xte, yte = build(te, limit)

    rf = RandomForestClassifier(n_estimators=500, random_state=SEED).fit(Xfit, yfit)
    order = np.argsort(rf.feature_importances_)[::-1][:8]
    print("  特徴量重要度 上位8:")
    for i in order:
        print(f"    {FEAT_NAMES[i]:>16}  {rf.feature_importances_[i]:.4f}")

    out = {"top_features": {FEAT_NAMES[i]: round(float(rf.feature_importances_[i]), 4)
                            for i in order}, "groups": {}}

    print("\n  特徴グループ別（そのグループだけで学習）:")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    Xall = np.vstack([Xfit, Xte])
    yall = np.concatenate([yfit, yte])
    for gname, idx in GROUPS.items():
        clf = RandomForestClassifier(n_estimators=500, random_state=SEED)
        acc = accuracy_score(yte, clf.fit(Xfit[:, idx], yfit).predict(Xte[:, idx]))
        sc = cross_val_score(RandomForestClassifier(n_estimators=500, random_state=SEED),
                             Xall[:, idx], yall, cv=cv, scoring="accuracy")
        print(f"    {gname:<34} test {acc * 100:6.2f}%   CV {sc.mean() * 100:5.2f}% ± {sc.std() * 100:.2f}%")
        out["groups"][gname] = {"test_acc": round(acc * 100, 2),
                                "cv_mean": round(float(sc.mean()) * 100, 2),
                                "cv_std": round(float(sc.std()) * 100, 2)}
    return out


MODELS = {
    "LogisticRegression": lambda: make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=5000, random_state=SEED)),
    "RandomForest": lambda: RandomForestClassifier(
        n_estimators=500, random_state=SEED),
    "GradientBoosting": lambda: GradientBoostingClassifier(random_state=SEED),
}


def main():
    tr, va, te = load_split("train"), load_split("valid"), load_split("test")
    print(f"split sizes: train={len(tr)} valid={len(va)} test={len(te)}")

    results = {"seq_len_conditions": {}}

    for tag, limit in [("first20 (= NetMamba と同条件)", SEQ_LEN), ("all packets", None)]:
        print(f"\n{'=' * 68}\n■ {tag}\n{'=' * 68}")
        Xtr, ytr = build(tr, limit)
        Xva, yva = build(va, limit)
        Xte, yte = build(te, limit)
        # train+valid で学習（NetMamba は valid を early-stop に使うだけなので概ね同等）
        Xfit = np.vstack([Xtr, Xva])
        yfit = np.concatenate([ytr, yva])

        dummy = DummyClassifier(strategy="most_frequent").fit(Xfit, yfit)
        base_acc = accuracy_score(yte, dummy.predict(Xte))
        print(f"  多数クラス基線 (test): {base_acc * 100:.2f}%")

        cond = {"majority_baseline": round(base_acc * 100, 2), "models": {}}

        # 全 504 会話での 5-fold CV（1 行 = 1 会話なので会話レベル CV と同義）
        Xall = np.vstack([Xfit, Xte])
        yall = np.concatenate([yfit, yte])
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

        for name, factory in MODELS.items():
            clf = factory().fit(Xfit, yfit)
            pred = clf.predict(Xte)
            acc = accuracy_score(yte, pred)
            f1 = f1_score(yte, pred)
            cm = confusion_matrix(yte, pred).tolist()
            scores = cross_val_score(factory(), Xall, yall, cv=cv, scoring="accuracy")

            print(f"\n  {name}")
            print(f"    test acc  : {acc * 100:.2f}%  ({int(acc * len(yte))}/{len(yte)})"
                  f"   F1={f1:.4f}   cm={cm}")
            print(f"    5-fold CV : {scores.mean() * 100:.2f}% ± {scores.std() * 100:.2f}%"
                  f"   (n=504)")
            cond["models"][name] = {
                "test_acc": round(acc * 100, 2),
                "test_f1": round(float(f1), 4),
                "confusion_matrix": cm,
                "cv_mean": round(float(scores.mean()) * 100, 2),
                "cv_std": round(float(scores.std()) * 100, 2),
            }

        results["seq_len_conditions"][tag] = cond

    results["attribution_first20"] = attribution(tr, va, te, SEQ_LEN)

    # ---- 単一特徴チェック：中央値 IAT だけの閾値分類（7/14 の主張の直接検証）
    print(f"\n{'=' * 68}\n■ 単一特徴：中央値 IAT のみ\n{'=' * 68}")
    med_tr = np.array([np.median(iats_of(it, SEQ_LEN)) for it in tr + va])
    med_te = np.array([np.median(iats_of(it, SEQ_LEN)) for it in te])
    y_tr = np.array([it["label"] for it in tr + va])
    y_te = np.array([it["label"] for it in te])

    for lbl, nm in [(0, "Direct"), (1, "Masque")]:
        v = med_tr[y_tr == lbl]
        print(f"  {nm:>6}: median-of-medians = {np.median(v) * 1e6:8.1f} µs   "
              f"(n={v.size}, q25={np.percentile(v, 25) * 1e6:.1f} q75={np.percentile(v, 75) * 1e6:.1f})")

    # train 上で最良閾値を選び test に適用（閾値選択に test を使わない）
    cands = np.unique(med_tr)
    best_t, best_a = None, -1.0
    for t in cands:
        a = max((med_tr > t).astype(int).__eq__(y_tr).mean(),
                (med_tr <= t).astype(int).__eq__(y_tr).mean())
        if a > best_a:
            best_a, best_t = a, t
    pred_gt = (med_te > best_t).astype(int)
    pred_le = (med_te <= best_t).astype(int)
    acc_thr = max(accuracy_score(y_te, pred_gt), accuracy_score(y_te, pred_le))
    print(f"\n  最良閾値 {best_t * 1e6:.1f} µs (train 上 {best_a * 100:.2f}%)"
          f" → test {acc_thr * 100:.2f}%")
    results["median_iat_threshold"] = {
        "threshold_us": round(float(best_t) * 1e6, 2),
        "train_acc": round(float(best_a) * 100, 2),
        "test_acc": round(float(acc_thr) * 100, 2),
    }

    # ---- 参考：NetMamba の符号化を通した後の数値差
    enc = lambda v: 1.0 / (1.0 + 1.0 / (1.0 + v))
    d_med = float(np.median(med_tr[y_tr == 0]))
    m_med = float(np.median(med_tr[y_tr == 1]))
    results["encoding_note"] = {
        "direct_median_us": round(d_med * 1e6, 2),
        "masque_median_us": round(m_med * 1e6, 2),
        "raw_ratio": round(m_med / max(d_med, 1e-12), 2),
        "encoded_direct": enc(d_med),
        "encoded_masque": enc(m_med),
        "encoded_gap": enc(m_med) - enc(d_med),
        "encoded_gap_as_fraction_of_range": (enc(m_med) - enc(d_med)) / 0.5,
    }
    e = results["encoding_note"]
    print(f"\n  符号化後: Direct {e['encoded_direct']:.9f} / Masque {e['encoded_masque']:.9f}")
    print(f"  → 生の比 {e['raw_ratio']}倍 が、符号化後は差 {e['encoded_gap']:.3e} "
          f"（値域の {e['encoded_gap_as_fraction_of_range'] * 100:.5f}%）")

    path = os.path.join(OUT, "summary_iat_baseline.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nsaved: {path}")


if __name__ == "__main__":
    main()
