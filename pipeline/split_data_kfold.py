"""[2026-08-23] E3: 把既有的单次 train/valid/test 切分改造成 5 折交叉验证的折数据集。

背景 (docs/01_Project_Management/E3_handoff.md):
论文里所有精度数字都撑在 51 条测试集上, 1 条错 = 1.96 pp。E3 用 5 折 out-of-fold
把它换成 504 条流的数字 + Wilson 置信区间 + 条件间配对检验 (McNemar)。

红线 (handoff §3.1): **五折的分配必须在所有条件间逐条相同**。
所以折分配只算一次, 落盘到 fold_assignment.json, 其余条件一律复用。
key 用 os.path.basename(pcap_file) —— 不同条件目录的 pcap_file 前缀不同, basename 相同。

红线 (handoff §3.2): 每折必须有独立的 valid (fine-tune.py 用它做模型选择/存 best ckpt),
绝不能拿该折的 test 充当 valid。valid 从该折的 train 部分里分层抽 10%, 种子固定。

注意 (handoff §3.4): 504 流 = 504 会话, 每会话恰好 1 条流, 所以 StratifiedGroupKFold
的分组约束实际为空, 与 StratifiedKFold 等价。这里仍显式传 groups 作防御性断言
(万一日后数据集变成一会话多流, 代码自动仍正确), 但**报告时不能暗示存在一个
实际不起作用的保护**。

    # 生成 assignment (只需一次) + 造 full 条件的 5 折
    python pipeline/split_data_kfold.py \
        --feat_dir data/processed/quic_vs_masque_bi_feat \
        --out_root data/processed/e3_folds/full

    # 其余条件复用同一份 assignment
    python pipeline/split_data_kfold.py \
        --feat_dir data/processed/bi_const_c123 \
        --out_root data/processed/e3_folds/c123
"""

import argparse
import json
import os
import sys

_ROOT = os.environ.get("MASQUE_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "pipeline"))

from split_data import session_key  # 复用会话解析, 不复制实现

DEFAULT_ASSIGNMENT = os.path.join(_ROOT, "outputs", "e3_kfold", "fold_assignment.json")


def flow_key(item):
    """折分配的 key。用 basename 而非绝对路径 —— 不同条件的 pcap_file 前缀不同。"""
    return os.path.basename(item["pcap_file"])


def load_items(feat_dir):
    """合并源目录的 train/valid/test, 按 flow_key 排序保证与目录内顺序无关。"""
    items = []
    for sp in ["train", "valid", "test"]:
        p = os.path.join(feat_dir, f"data-{sp}.json")
        with open(p, encoding="utf-8") as f:
            items += json.load(f)
    keys = [flow_key(it) for it in items]
    if len(set(keys)) != len(keys):
        raise SystemExit(f"basename 不唯一, 无法作为折分配的 key: {feat_dir}")
    items.sort(key=flow_key)  # 排序 => 分配与源目录内的记录顺序无关
    return items


def build_assignment(items, n_splits, valid_ratio, seed):
    from sklearn.model_selection import StratifiedGroupKFold, train_test_split

    y = [it["label"] for it in items]
    groups = [str(session_key(it)) for it in items]
    keys = [flow_key(it) for it in items]

    n_sessions = len(set(groups))
    print(f"flows {len(items)}, sessions {n_sessions} "
          f"({len(items)/n_sessions:.2f} flows/session)")
    if n_sessions == len(items):
        print("注: 每会话恰好 1 条流 -> group 约束为空, 等价于 StratifiedKFold "
              "(显式传 groups 仅作防御性断言, 见 handoff §3.4)")

    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    test_fold = {}
    valid_in_fold = {}
    for k, (rest_idx, test_idx) in enumerate(skf.split(items, y, groups=groups)):
        for i in test_idx:
            test_fold[keys[i]] = k
        # 该折的非 test 部分里, 按 label 分层抽 valid_ratio 作 valid, 其余作 train
        rest_y = [y[i] for i in rest_idx]
        tr_idx, va_idx = train_test_split(
            list(rest_idx), test_size=valid_ratio, stratify=rest_y,
            random_state=seed * 1000 + k, shuffle=True)
        valid_in_fold[str(k)] = sorted(keys[i] for i in va_idx)
        print(f"  fold {k}: test {len(test_idx)}, valid {len(va_idx)}, train {len(tr_idx)}")

    return {
        "n_splits": n_splits,
        "valid_ratio": valid_ratio,
        "seed": seed,
        "splitter": "StratifiedGroupKFold(shuffle=True)",
        "n_flows": len(items),
        "n_sessions": n_sessions,
        "key": "os.path.basename(pcap_file)",
        "test_fold": test_fold,
        "valid_in_fold": valid_in_fold,
    }


def write_folds(items, assign, out_root):
    n_splits = assign["n_splits"]
    test_fold = assign["test_fold"]
    by_key = {flow_key(it): it for it in items}

    missing = set(test_fold) - set(by_key)
    extra = set(by_key) - set(test_fold)
    if missing or extra:
        raise SystemExit(f"与 assignment 不匹配: 缺 {len(missing)} 条, 多 {len(extra)} 条\n"
                         f"  缺例: {sorted(missing)[:3]}\n  多例: {sorted(extra)[:3]}")

    for k in range(n_splits):
        valid_keys = set(assign["valid_in_fold"][str(k)])
        tr, va, te = [], [], []
        for key, it in by_key.items():
            if test_fold[key] == k:
                te.append(it)
            elif key in valid_keys:
                va.append(it)
            else:
                tr.append(it)
        assert len(tr) + len(va) + len(te) == len(items), "三集合之和 != 全集"
        assert va, f"fold {k} 的 valid 为空"

        d = os.path.join(out_root, f"fold{k}")
        os.makedirs(d, exist_ok=True)
        for name, part in [("train", tr), ("valid", va), ("test", te)]:
            # 排序保证可复现; test 的顺序即 dump 预测的顺序 (SequentialSampler)
            part = sorted(part, key=flow_key)
            with open(os.path.join(d, f"data-{name}.json"), "w", encoding="utf-8") as f:
                json.dump(part, f, indent=2)
        print(f"  fold{k}: train {len(tr)} / valid {len(va)} / test {len(te)} -> {d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_dir", required=True, help="源数据集目录 (含 data-{train,valid,test}.json)")
    ap.add_argument("--out_root", required=True, help="折数据集输出根目录, 内含 fold0..fold4")
    ap.add_argument("--assignment", default=DEFAULT_ASSIGNMENT,
                    help="折分配 JSON; 不存在则生成, 存在则复用 (所有条件必须复用同一份)")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--valid_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    feat_dir = args.feat_dir if os.path.isabs(args.feat_dir) else os.path.join(_ROOT, args.feat_dir)
    out_root = args.out_root if os.path.isabs(args.out_root) else os.path.join(_ROOT, args.out_root)
    apath = args.assignment if os.path.isabs(args.assignment) else os.path.join(_ROOT, args.assignment)

    items = load_items(feat_dir)

    if os.path.exists(apath):
        with open(apath, encoding="utf-8") as f:
            assign = json.load(f)
        print(f"复用已有折分配: {apath} (n_splits={assign['n_splits']}, seed={assign['seed']})")
    else:
        print(f"生成折分配: {apath}")
        assign = build_assignment(items, args.n_splits, args.valid_ratio, args.seed)
        os.makedirs(os.path.dirname(apath), exist_ok=True)
        with open(apath, "w", encoding="utf-8") as f:
            json.dump(assign, f, indent=2)

    os.makedirs(out_root, exist_ok=True)
    write_folds(items, assign, out_root)

    # metadata.json 从源目录复制 (内容为 {"name_to_idx": {...}})
    with open(os.path.join(feat_dir, "metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    for k in range(assign["n_splits"]):
        with open(os.path.join(out_root, f"fold{k}", "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    print(f"完成: {out_root} (metadata: {meta})")


if __name__ == "__main__":
    main()
