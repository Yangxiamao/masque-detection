import os
import json
import random
from collections import defaultdict

# ---------------------------------------------------------------------------
# [2026-07-13] 修复会话级数据泄漏 (路线A)
# 原实现按 label 分组后对"单条流"random.shuffle 切分，导致同一 TLS 会话被
# split_quic_vs_masque.py 拆出的上/下行两条单向流可能落入不同集合 (train/test)，
# 构成会话级数据泄漏 (旧切分中 71.4% 的测试样本的另一方向在训练集)。
# 现改为【会话级切分】: 先按会话归组(上/下行归一为同一会话)，再对"会话"切分，
# 保证同一会话的所有流始终落入同一集合。
# ---------------------------------------------------------------------------


def session_key(item):
    """从 pcap_file 文件名解析规范化会话ID，使上/下行两条单向流归一为同一会话。
    文件名形如: class1_Masque_iter1_20260616_011456_<ip1>_<port1>_<ip2>_<port2>
    IPv4 各占4段、port 各占1段 => 末尾10段为五元组，其余为 batch 前缀(含iter与时间戳)。
    会话ID = (batch前缀, 排序后的两端点)。对端点排序即可让方向无关。"""
    name = os.path.basename(item["pcap_file"]).replace(".pcap", "")
    segs = name.split("_")
    quint = segs[-10:]
    batch = "_".join(segs[:-10])
    ip1 = ".".join(quint[0:4]); p1 = quint[4]
    ip2 = ".".join(quint[5:9]); p2 = quint[9]
    endpoints = tuple(sorted([(ip1, p1), (ip2, p2)]))
    return (batch, endpoints)


def split_data(input_file, output_dir, train_ratio=0.8, valid_ratio=0.1):
    print(f"Loading data from {input_file}...")
    with open(input_file, "r") as f:
        data = json.load(f)

    print(f"Total flows: {len(data)}")

    name_to_idx = {}
    for item in data:
        name_to_idx.setdefault(item["name"], item["label"])

    # 1) 按会话归组 (上/下行归一)
    sessions = defaultdict(list)
    for item in data:
        sessions[session_key(item)].append(item)
    print(f"Independent sessions: {len(sessions)} "
          f"(avg {len(data)/len(sessions):.2f} flows/session)")

    # 2) 按 label 分组会话ID，保证分层比例
    label_sessions = defaultdict(list)
    for sid, items in sessions.items():
        label_sessions[items[0]["label"]].append(sid)

    train_data, valid_data, test_data = [], [], []

    random.seed(42)  # fixed seed for reproducible split

    # 3) 对"会话"而非"流"进行切分
    for label, sids in label_sessions.items():
        random.shuffle(sids)
        total = len(sids)
        train_end = int(total * train_ratio)
        valid_end = train_end + int(total * valid_ratio)

        tr_sids = sids[:train_end]
        va_sids = sids[train_end:valid_end]
        te_sids = sids[valid_end:]

        for sid in tr_sids:
            train_data.extend(sessions[sid])
        for sid in va_sids:
            valid_data.extend(sessions[sid])
        for sid in te_sids:
            test_data.extend(sessions[sid])

        lname = [k for k, v in name_to_idx.items() if v == label][0]
        print(f"Label {label} ({lname}): {total} sessions -> "
              f"train {len(tr_sids)}, valid {len(va_sids)}, test {len(te_sids)} sessions")

    # 打乱最终列表 (流级)
    random.shuffle(train_data)
    random.shuffle(valid_data)
    random.shuffle(test_data)

    os.makedirs(output_dir, exist_ok=True)

    print(f"Writing train set: {len(train_data)} flows...")
    with open(os.path.join(output_dir, "data-train.json"), "w") as f:
        json.dump(train_data, f, indent=2)

    print(f"Writing valid set: {len(valid_data)} flows...")
    with open(os.path.join(output_dir, "data-valid.json"), "w") as f:
        json.dump(valid_data, f, indent=2)

    print(f"Writing test set: {len(test_data)} flows...")
    with open(os.path.join(output_dir, "data-test.json"), "w") as f:
        json.dump(test_data, f, indent=2)

    metadata = {"name_to_idx": name_to_idx}
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {os.path.join(output_dir, 'metadata.json')}")
    print("Done!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--feat_dir", type=str, required=True, help="Directory containing data.json")
    args = parser.parse_args()

    input_file = os.path.join(args.feat_dir, "data.json")
    split_data(input_file, args.feat_dir)
