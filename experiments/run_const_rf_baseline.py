"""[2026-08-18] E1-0: 三个配置常量的带方差数字 + 消融是否真的生效的经典分类器对照。

为什么需要这个 (交接文档 §4.4):
  "有 Retry" 单规则的 97.82% 是**在全部 504 条流上算的描述性数字**, 没做交叉验证,
  不可与 NetMamba+ 的 test 数字并列。本脚本给出可对外引用的带方差版本。

为什么还要 Part B/C (CLAUDE.md 规则 + 交接文档 §9.2):
  "消除某特征后精度不变" 的结果必须先证明该特征真的被消除了, 而不是相信代码逻辑。
  7/20 曾因一个未察觉的编码缺陷同时得出三个错误结论。

Part A  三常量 (max / min / has_retry) 的随机森林 + 会话级 5-fold, n=504
Part B  三条零训练规则在 orig 与 C123 的 signed_sizes 上各自的准确率 -> 应崩到近随机
Part C  只用长度通道摘要统计的随机森林, orig vs C123 -> 常量抹掉后长度还剩多少信号

    wsl -e bash -lic 'conda activate MAMBA; cd /mnt/d/zemi/2026/dataset; \
        python experiments/run_const_rf_baseline.py'
"""
import json
import os
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score

_ROOT = os.environ.get("MASQUE_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(_ROOT, "experiments"))
sys.path.append(os.path.join(_ROOT, "pipeline"))

from analyze_header_geometry import collect          # noqa: E402
from split_data import session_key                   # noqa: E402

OUT = os.path.join(_ROOT, "outputs", "const_rf_baseline")
SEED = 42
N_SPLITS = 5

# signed_sizes 尺度 = UDP 载荷 + 42 (以太网帧)
ETH = 42
SPLITS = ("train", "valid", "test")


def cv(X, y, groups, tag):
    """会话级分层 5-fold。1 流 = 1 会话, 但仍显式按 session_key 分组。"""
    skf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    clf = RandomForestClassifier(n_estimators=500, random_state=SEED)
    s = cross_val_score(clf, X, y, groups=groups, cv=skf, scoring="accuracy")
    print(f"  {tag:<46}{s.mean()*100:6.2f}% ± {s.std()*100:.2f}%   "
          f"folds={[round(v*100, 2) for v in s]}")
    return {"cv_mean": round(float(s.mean()) * 100, 2),
            "cv_std": round(float(s.std()) * 100, 2),
            "folds": [round(float(v) * 100, 2) for v in s]}


# --------------------------------------------------------------------------
# Part A: 三常量 (来自 pcap 解析, UDP 载荷尺度)
# --------------------------------------------------------------------------

def part_a():
    print("=" * 84)
    print("Part A  三个配置常量的随机森林 + 会话级 5-fold (n=504, UDP 载荷尺度)")
    print("=" * 84)
    rows = [r for r in collect() if r["up"] + r["dn"]]
    X, y, g = [], [], []
    for r in rows:
        pkts = r["up"] + r["dn"]
        X.append([max(pkts), min(pkts),
                  float(any(c["ptype"] == 3 for c in r["cids"]))])
        y.append(1 if r["cls"] == "Masque" else 0)
        g.append(str(session_key({"pcap_file": r["file"]})))
    X, y, g = np.asarray(X, float), np.asarray(y), np.asarray(g)
    print(f"  n={len(y)}  Direct={int((y == 0).sum())}  Masque={int((y == 1).sum())}  "
          f"独立会话={len(set(g))}")
    maj = max((y == 0).mean(), (y == 1).mean())
    print(f"  多数类基线 {maj*100:.2f}%\n")

    res = {"n": int(len(y)), "n_groups": len(set(g)),
           "majority_baseline": round(float(maj) * 100, 2), "cv": {}}
    res["cv"]["三常量合用 (max, min, has_retry)"] = cv(X, y, g, "三常量合用 (max, min, has_retry)")
    for i, nm in enumerate(["仅 max_packet", "仅 min_packet", "仅 has_retry"]):
        res["cv"][nm] = cv(X[:, [i]], y, g, nm)
    res["cv"]["max + min (不含 Retry)"] = cv(X[:, [0, 1]], y, g, "max + min (不含 Retry)")
    return res


# --------------------------------------------------------------------------
# Part B / C: 直接读特征 JSON, 检验消融是否生效
# --------------------------------------------------------------------------

def load_feat(dirname):
    items = []
    for sp in SPLITS:
        with open(os.path.join(_ROOT, "data", "processed", dirname,
                               f"data-{sp}.json"), encoding="utf-8") as f:
            items += json.load(f)
    return items


def sizes_of(it):
    return [v for v in (int(x) for x in it["signed_sizes"].split()) if v != 0]


def part_b(retry_map):
    print()
    print("=" * 84)
    print("Part B  三条零训练规则在 signed_sizes 上的准确率: orig vs C123 (n=504)")
    print("=" * 84)
    print("  (规则阈值按帧尺度换算: 1200->1242, 42->84; has_retry 取真实 Retry 帧是否仍在流内)")
    print()
    res = {}
    for tag, dirname in (("orig", "quic_vs_masque_bi_feat"), ("C123", "bi_const_c123")):
        items = load_feat(dirname)
        y = np.asarray([it["label"] for it in items])
        vals = [sizes_of(it) for it in items]
        base = os.path.basename
        n_pkt = np.asarray([it["num_packet"] for it in items])
        n_orig = np.asarray([retry_map[base(it["pcap_file"].replace("\\", "/"))]["n_frames"]
                             for it in items])
        preds = {
            "max == 1242 (原 1200)": np.asarray([max(map(abs, v)) == 1200 + ETH for v in vals]),
            "min >= 84   (原 42)":   np.asarray([min(map(abs, v)) >= 42 + ETH for v in vals]),
            "流内仍含 Retry 帧":       n_pkt == n_orig if dirname != "quic_vs_masque_bi_feat"
                                     else np.asarray([len(retry_map[base(it["pcap_file"].replace("\\", "/"))]["retry_idx"]) > 0
                                                      for it in items]),
        }
        if dirname == "bi_const_c123":
            # C123 里 Retry 帧已被剔除, 故"含 Retry"恒为假; 用"包数是否少于原 pcap"作为
            # 该规则在消融后还剩多少辨别力的等价检验 (即攻击者能否从缺包数反推)
            preds["流内仍含 Retry 帧"] = n_pkt < n_orig
        res[tag] = {}
        print(f"  --- {tag}")
        for nm, p in preds.items():
            acc = max((p == (y == 1)).mean(), (p == (y == 0)).mean())
            tpr = p[y == 1].mean()
            fpr = p[y == 0].mean()
            print(f"      {nm:<26} acc={acc*100:6.2f}%  TPR={tpr*100:5.1f}%  FPR={fpr*100:5.1f}%")
            res[tag][nm] = {"acc": round(float(acc) * 100, 2),
                            "tpr": round(float(tpr) * 100, 1),
                            "fpr": round(float(fpr) * 100, 1)}
        print()
    return res


def len_features(it):
    v = sizes_of(it)
    up = [abs(x) for x in v if x > 0]
    dn = [abs(x) for x in v if x < 0]
    f = []
    for a in (list(map(abs, v)), up, dn):
        a = a or [0]
        f += [max(a), min(a), float(np.mean(a)), float(np.std(a)),
              float(np.percentile(a, 25)), float(np.median(a)),
              float(np.percentile(a, 75)), len(a)]
    return f


def part_c(kinds):
    print("=" * 84)
    print("Part C  只用长度通道摘要统计的随机森林 + 会话级 5-fold (n=504)")
    print("=" * 84)
    print("  常量抹掉后长度通道还剩多少可分性 —— 用来判断消融是否只是把常量换了个位置。")
    print()
    res = {}
    for tag, dirname in kinds:
        items = load_feat(dirname)
        X = np.asarray([len_features(it) for it in items], float)
        y = np.asarray([it["label"] for it in items])
        g = np.asarray([str(session_key(it)) for it in items])
        res[tag] = cv(X, y, g, tag)
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(_ROOT, "outputs", "header_geometry", "retry_index.json"),
              encoding="utf-8") as f:
        retry_map = json.load(f)

    out = {"part_a_three_constants": part_a(),
           "part_b_unfitted_rules": part_b(retry_map),
           "part_c_length_channel_rf": part_c([
               ("orig 全特征长度通道", "quic_vs_masque_bi_feat"),
               ("C1  MTU 归一化",      "bi_const_c1"),
               ("C2  CID 归一化",      "bi_const_c2"),
               ("C3  去 Retry",        "bi_const_c3"),
               ("C123 三者同时",       "bi_const_c123"),
           ])}

    path = os.path.join(OUT, "summary_const_rf_baseline.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nsaved: {path}")


if __name__ == "__main__":
    main()
