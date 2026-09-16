import os
import scapy.all as scapy
from collections import defaultdict
import time

# 仓库根:优先取 MASQUE_ROOT 环境变量,否则由本文件位置推导。
# 如此 git worktree 中的副本会各自指向自己的目录,输出互不覆盖。
_ROOT = os.environ.get("MASQUE_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# [2026-07-13] 路线B: 双向流切分.
# 修复 split_quic_vs_masque.py:43 的方向敏感 flow_key 缺陷:
# 改用【规范化五元组】(对两端点排序) 作为 key, 使同一会话的上/下行合并为一条
# 双向流; 保存前按时间戳排序, 保留真实的 client<->server 交互序列.
# 输出目录: data/processed/quic_vs_masque_bi/{Direct,Masque}

RAW_DIR = os.path.join(_ROOT, "data", "raw")
OUT_DIR = os.path.join(_ROOT, "data", "processed", "quic_vs_masque_bi")
MAX_PACKETS_PER_FLOW = 50
MIN_PACKETS_PER_FLOW = 50  # 仅保存达到完整长度的流, 保证数据一致


def norm_flow_key(src_ip, sport, dst_ip, dport):
    """规范化五元组: 对两个端点排序, 使上/下行得到同一 key (方向无关)。"""
    ep1 = (src_ip, sport)
    ep2 = (dst_ip, dport)
    a, b = sorted([ep1, ep2])
    return f"{a[0]}_{a[1]}_{b[0]}_{b[1]}"


def process_pcap(filepath, out_folder, prefix):
    print(f"Processing {filepath} -> {out_folder}")
    os.makedirs(out_folder, exist_ok=True)

    flow_dict = defaultdict(list)
    flow_completed = set()

    start_time = time.time()
    with scapy.PcapReader(filepath) as pcap_reader:
        idx = 0
        while True:
            try:
                pkt = pcap_reader.read_packet()
                if pkt is None:
                    break
            except EOFError:
                break
            except Exception:
                continue

            idx += 1
            if idx % 50000 == 0:
                print(f"Read {idx} packets... ({time.time() - start_time:.2f}s)")

            if pkt.haslayer("UDP") and pkt.haslayer("IP"):
                sport = pkt["UDP"].sport
                dport = pkt["UDP"].dport
                if sport == 443 or dport == 443:
                    src_ip = pkt["IP"].src
                    dst_ip = pkt["IP"].dst
                    flow_key = norm_flow_key(src_ip, sport, dst_ip, dport)

                    if flow_key in flow_completed:
                        continue

                    flow_dict[flow_key].append(pkt)

                    if len(flow_dict[flow_key]) >= MAX_PACKETS_PER_FLOW:
                        flow_completed.add(flow_key)

    saved_count = 0
    for flow_key, packets in flow_dict.items():
        if len(packets) >= MIN_PACKETS_PER_FLOW:
            # 按时间戳排序, 确保真实的双向交互时序 (双向合并后尤其重要)
            packets = sorted(packets, key=lambda p: float(p.time))
            safe_key = flow_key.replace(":", "_").replace(".", "_")
            out_path = os.path.join(out_folder, f"{prefix}_{safe_key}.pcap")
            scapy.wrpcap(out_path, packets)
            saved_count += 1

    print(f"Saved {saved_count} bi-flows from {filepath}")


def main():
    direct_files = [f for f in os.listdir(RAW_DIR) if f.startswith("class0_Direct") and f.endswith(".pcap")]
    masque_files = [f for f in os.listdir(RAW_DIR) if f.startswith("class1_Masque") and f.endswith(".pcap")]

    direct_out = os.path.join(OUT_DIR, "Direct")
    for f in direct_files:
        process_pcap(os.path.join(RAW_DIR, f), direct_out, f.replace(".pcap", ""))

    masque_out = os.path.join(OUT_DIR, "Masque")
    for f in masque_files:
        process_pcap(os.path.join(RAW_DIR, f), masque_out, f.replace(".pcap", ""))


if __name__ == "__main__":
    main()
