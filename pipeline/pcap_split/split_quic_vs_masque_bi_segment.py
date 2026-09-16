"""[2026-09-08] E6-C: 取流的【任意一段】而不只是开头 50 包。

## 动机: 分离"隧道建立"与"隧道运行"

现有管道 (split_quic_vs_masque_bi.py) 取每条流的【前 50 个包】。这与两件事叠在一起:

  1. 采集程序在每个 MASQUE 域名前起隧道、访问完拆隧道 (5 秒空闲等待), Direct 侧
     没有对应操作 —— 即每条 MASQUE 流都带着一次全新的隧道建立, 而 Direct 流不带
     (论文 Limitations 已承认, 但没有任何分析去掉它);
  2. E1 的 P-C 发现 Retry 地址验证出现在 259/270 条 MASQUE 流、0/234 条 Direct 流。

于是一个刻薄但完全合理的审稿意见是: 分类器可能主要在区分【冷启动的新建连接】与
【已经握完手的连接】, 而不是【隧道】与【非隧道】。C123 消融部分回应了这一点,
但 C3 只删掉 Retry 帧, 删不掉它引入的那个往返。

本脚本给出直接的检验: 丢掉前 skip 个包, 取接下来的 take 个包 —— 一个完全处于
稳态传输阶段、不含任何握手的窗口。若稳态段仍然可分, 上述解释就被排除。

必须成对使用的对照: 同一批流的【前 50 包】(--skip 0)。因为 >= skip+take 包的
过滤会改变流的集合 (448 vs 504 条), 不控制这一点就分不清"精度变化"来自
"换了流的哪一段"还是"换了哪些流"。

    # 稳态段 (第 51-100 包) 与其头部对照, 用同一批 >= 100 包的流
    python pipeline/pcap_split/split_quic_vs_masque_bi_segment.py --skip 50 --take 50
    python pipeline/pcap_split/split_quic_vs_masque_bi_segment.py --skip 0  --take 50 \
        --min-packets 100 --out-suffix head448
"""

import argparse
import os
import time
from collections import defaultdict

import scapy.all as scapy

_ROOT = os.environ.get("MASQUE_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(_ROOT, "data", "raw")


def norm_flow_key(src_ip, sport, dst_ip, dport):
    """与 split_quic_vs_masque_bi.py 逐字相同的规范化五元组。"""
    a, b = sorted([(src_ip, sport), (dst_ip, dport)])
    return f"{a[0]}_{a[1]}_{b[0]}_{b[1]}"


def process_pcap(filepath, out_folder, prefix, skip, take, min_packets):
    print(f"Processing {filepath} -> {out_folder}", flush=True)
    os.makedirs(out_folder, exist_ok=True)

    need = max(min_packets, skip + take)
    flow_dict = defaultdict(list)
    flow_completed = set()

    t0 = time.time()
    with scapy.PcapReader(filepath) as reader:
        idx = 0
        while True:
            try:
                pkt = reader.read_packet()
                if pkt is None:
                    break
            except EOFError:
                break
            except Exception:
                continue

            idx += 1
            if idx % 200000 == 0:
                print(f"  read {idx} packets ({time.time()-t0:.0f}s), "
                      f"{len(flow_completed)}/{len(flow_dict)} flows complete", flush=True)

            if not (pkt.haslayer("UDP") and pkt.haslayer("IP")):
                continue
            sport, dport = pkt["UDP"].sport, pkt["UDP"].dport
            if sport != 443 and dport != 443:
                continue

            key = norm_flow_key(pkt["IP"].src, sport, pkt["IP"].dst, dport)
            if key in flow_completed:
                continue
            flow_dict[key].append(pkt)
            if len(flow_dict[key]) >= need:
                flow_completed.add(key)

    saved = 0
    for key, packets in flow_dict.items():
        if len(packets) < need:
            continue
        packets = sorted(packets, key=lambda p: float(p.time))
        seg = packets[skip:skip + take]
        if len(seg) < take:
            continue
        safe = key.replace(":", "_").replace(".", "_")
        scapy.wrpcap(os.path.join(out_folder, f"{prefix}_{safe}.pcap"), seg)
        saved += 1

    print(f"  saved {saved} flows (segment [{skip}:{skip+take}], "
          f"need >= {need} packets)  {time.time()-t0:.0f}s", flush=True)
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", type=int, default=50, help="丢弃开头多少个包")
    ap.add_argument("--take", type=int, default=50, help="取多少个包 (与主管道一致 = 50)")
    ap.add_argument("--min-packets", type=int, default=None,
                    help="流的最小包数; 默认 = skip+take。对照组需显式设成与稳态组相同")
    ap.add_argument("--out-suffix", default=None, help="输出目录后缀; 默认 seg{skip}_{take}")
    args = ap.parse_args()

    min_packets = args.min_packets if args.min_packets is not None else args.skip + args.take
    suffix = args.out_suffix or f"seg{args.skip}_{args.take}"
    out_dir = os.path.join(_ROOT, "data", "processed", f"quic_vs_masque_bi_{suffix}")
    print(f"输出: {out_dir}  (skip={args.skip} take={args.take} min={min_packets})\n")

    total = {}
    for cls, pat, sub in [("Direct", "class0_Direct", "Direct"),
                          ("MASQUE", "class1_Masque", "Masque")]:
        files = sorted(f for f in os.listdir(RAW_DIR)
                       if f.startswith(pat) and f.endswith(".pcap"))
        n = 0
        for f in files:
            n += process_pcap(os.path.join(RAW_DIR, f),
                              os.path.join(out_dir, sub), f.replace(".pcap", ""),
                              args.skip, args.take, min_packets)
        total[cls] = n

    print(f"\n完成: " + "  ".join(f"{k} {v}" for k, v in total.items()))


if __name__ == "__main__":
    main()
