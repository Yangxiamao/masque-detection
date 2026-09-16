import os
import sys

# 把 dataset_scripts 目录加入 path 以导入公共模块
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "NetMambaPlus", "dataset_scripts"))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# [2026-07-13] 路线B: 双向特征提取.
# 关键点1: 使用 dataset_mm_bi_common.pcap_to_pt_json, 输出带方向的 signed_sizes 字段
#          (正=forward client->server, 负=backward server->client)。
# 关键点2: 去偏必须【保留方向】。原 uni 管道用 zero_ip_port 把 src/dst IP 全归零,
#          会让 bi 管道的方向判断(src_ip==first_src_ip)全部误判为 forward。
#          故改用 [relative_zero_ip, zero_port]: relative_zero_ip 在归零前先按真实 IP
#          判定方向并映射 forward->0.0.0.0 / backward->0.0.0.1, 既隐藏真实 IP 又保留方向;
#          zero_port 再抹掉端口。
from dataset_json_common import relative_zero_ip, zero_port
from dataset_mm_bi_common import pcap_to_pt_json

# 仓库根:优先取 MASQUE_ROOT 环境变量,否则由本文件位置推导。
# 如此 git worktree 中的副本会各自指向自己的目录,输出互不覆盖。
_ROOT = os.environ.get("MASQUE_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 与 baseline (feat_quic_vs_masque.py) 保持一致
BYTE_PACKET_NUM = 5
SEQ_PACKET_NUM = 50
HEADER_BYTES = 80
PAYLOAD_BYTES = 240


def main():
    # [2026-09-08] 加了两个可选参数, 默认值与原来逐字相同 —— 稳态段实验 (E6-C)
    # 需要对另一个切流目录跑同一套特征提取, 不能靠改写死路径来做。
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcap_dir", default="quic_vs_masque_bi",
                    help="data/processed/ 下的切流目录名")
    ap.add_argument("--save_dir", default="quic_vs_masque_bi_feat",
                    help="data/processed/ 下的特征输出目录名")
    args = ap.parse_args()

    pcap_dir = os.path.join(_ROOT, "data", "processed", args.pcap_dir)
    save_dir = os.path.join(_ROOT, "data", "processed", args.save_dir)

    debias_strategy = [relative_zero_ip, zero_port]  # 保留方向的去偏

    print(f"Extracting BI features from {pcap_dir} -> {save_dir}")
    pcap_to_pt_json(
        pcap_dirs=[pcap_dir],
        save_dir=save_dir,
        filter_func=None,
        byte_packet_num=BYTE_PACKET_NUM,
        seq_packet_num=SEQ_PACKET_NUM,
        multi_chunk=False,
        header_bytes=HEADER_BYTES,
        payload_bytes=PAYLOAD_BYTES,
        debias_funcs=debias_strategy,
        merge_packet_bytes=True,
    )
    print("BI feature extraction complete!")


if __name__ == "__main__":
    main()
