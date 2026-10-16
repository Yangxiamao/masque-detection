# -*- coding: utf-8 -*-
"""[2026-10-16 ICOIN r4 #1] 方向符号按【真实 client 地址】锚定, 而不是按"窗口第一个包的发送方"。

## 缺陷

NetMambaPlus/dataset_scripts 里有两处都把 packets[0] 的发送方当作 forward (正号):
  * dataset_json_common.relative_zero_ip: forward -> 0.0.0.0 / backward -> 0.0.0.1 哨兵
  * dataset_mm_bi_common.get_chunk_feature: signed_sizes 的正负号
从会话开头切的窗口 (主数据 504 条、head448 对照) 第一个包恒为 client Initial, 两者恰好等于
论文 III-B 的定义 "正 = client->server" (已核实 504/504, 448/448)。但 E6 稳态段把第 51-100 包
另存为 pcap 再走同一管道, 锚点变成第 51 包, 448 条里 246 条是 server
(Direct 123/189, MASQUE 123/259) —— 符号与 III-B 相反, 且翻转率与类别相关。

## 修法

client = UDP 端口不是 443 的一侧 (切流只保留 sport/dport 含 443 的流, 服务端恒为 443)。
  * client_zero_ip: relative_zero_ip 的 client 锚定版 (哨兵映射与原函数逐字相同)
  * get_chunk_feature_client: 去偏后 client 恒为 0.0.0.0 / ::, 以此判定正号
install() 把后者打补丁进 dataset_mm_bi_common (get_flow_feature 按模块全局名调用它)。

原本第一个包就是 client 的窗口, 输出与旧管道逐字节相同 (run_steady_client_anchor.py 有核对)。
不改 NetMambaPlus submodule。
"""

import dataset_mm_bi_common as _bi

CLIENT_V4, SERVER_V4 = "0.0.0.0", "0.0.0.1"
CLIENT_V6, SERVER_V6 = "0:0:0:0:0:0:0:0", "0:0:0:0:0:0:0:1"
SERVER_PORT = 443


def _ip(packet):
    return packet["IPv6"] if packet.haslayer("IPv6") else packet["IP"]


def find_client_ip(packets):
    for packet in packets:
        if not packet.haslayer("UDP") or not (packet.haslayer("IP") or packet.haslayer("IPv6")):
            continue
        sport, dport = packet["UDP"].sport, packet["UDP"].dport
        if dport == SERVER_PORT and sport != SERVER_PORT:
            return _ip(packet).src
        if sport == SERVER_PORT and dport != SERVER_PORT:
            return _ip(packet).dst
    raise ValueError("无法判定 client: 流中没有恰好一端为 443 的 UDP 包")


def client_zero_ip(packets):
    """必须在 zero_port 之前调用 (要读端口)。"""
    client_ip = find_client_ip(packets)
    for packet in packets:
        if packet.haslayer("IP"):
            if packet["IP"].src == client_ip:
                packet["IP"].src, packet["IP"].dst = CLIENT_V4, SERVER_V4
            else:
                packet["IP"].src, packet["IP"].dst = SERVER_V4, CLIENT_V4
        if packet.haslayer("IPv6"):
            if packet["IPv6"].src == client_ip:
                packet["IPv6"].src, packet["IPv6"].dst = CLIENT_V6, SERVER_V6
            else:
                packet["IPv6"].src, packet["IPv6"].dst = SERVER_V6, CLIENT_V6
    return packets


def get_chunk_feature_client(packets, byte_packet_num, header_bytes, payload_bytes,
                             merge_packet_bytes):
    """与 dataset_mm_bi_common.get_chunk_feature 相同, 仅正号改为 "src 是 client 哨兵"。"""
    bytes_list = []
    for packet in packets[:byte_packet_num]:
        header_arr, payload_arr = _bi.get_packet_feature(packet, header_bytes, payload_bytes,
                                                         merge_packet_bytes)
        if merge_packet_bytes:
            bytes_list.append(header_arr + payload_arr)
        else:
            bytes_list.append([header_arr, payload_arr])
    signed_sizes, intervals = [], []
    for idx, packet in enumerate(packets):
        if _ip(packet).src in (CLIENT_V4, CLIENT_V6):
            signed_sizes.append(len(packet))     # + client -> server
        else:
            signed_sizes.append(-len(packet))    # - server -> client
        if idx == 0:
            intervals.append(0)
        else:
            intervals.append(round(float(packet.time - packets[idx - 1].time), 6))
    assert len(signed_sizes) == len(intervals)
    return bytes_list, signed_sizes, intervals


def install():
    _bi.get_chunk_feature = get_chunk_feature_client
