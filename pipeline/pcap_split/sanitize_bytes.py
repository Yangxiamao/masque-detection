"""[2026-10-12 ICOIN r2 #1] 字节通道去长度 / 去头区方向: tf_sanitize_bytes + 自检。

背景 (docs/01_Planning/Handoff_ICOIN_Revision_2026-10-12.md §2-2, §3 #1):
v1 去偏后的 28 B IP+UDP 头里, 除了 IP/UDP 长度字段, TTL (上行 128 / 下行 56-57)、
IP ID (上行随机 / 下行几乎全 0)、IP 校验和 (上行全 0 / 下行有值)、UDP 校验和、
DSCP/ECN、flags (仅 Direct 下行有 2 种取值) 都按方向或类别取不同值
(outputs/v3_kfold/header_fields_by_dir.json 的 v1 表)。载荷区里还有三处藏着包长:
QUIC 长包头的 Length 字段、合并包 (coalesced) 里第二个包头的位置、以及载荷不足
240 B 时补的零。

变换 (每包 320 B = 80 B 头区 + 240 B 载荷区, 载荷区从 UDP 载荷第 0 字节开始):
  头区   : 整体替换为固定模板 (ver/IHL=0x45, flags=0x40, proto=17, 其余全 0)。
           等价于 handoff 的白名单, 只是把白名单字段也钉成常量 —— Direct 下行的
           flags 有 2 种取值, 保留原值会让"两类取值集合相同"不成立。
  载荷区 : 先由原始 IP total length 得 UDP 载荷长 L = totlen - 28, 再按包类型:
    - QUIC 长包头且带 Length (Initial / 0-RTT / Handshake):
        保留 [0, Length 字段末尾), 其中 Length 字段清零; 之后全部填伪随机字节。
        (之后是受保护的包号 + 密文, 本就与随机字节不可区分; 一并替换顺便抹掉
         合并包里第二个包头的位置。)
    - Retry / Version Negotiation / 无法解析的长包头:
        保留到 SCID 末尾 (或能解析到的位置), 之后填伪随机。
    - 短包头 / 非 QUIC: 保留 [0, min(L, 240)), 补零区 [L, 240) 填伪随机。
  伪随机字节按 pcap 文件名派生 (与 simulate_obfuscation_bi._flow_rng 同法), 逐条可复现。

⚠️ 本变换【不】去除 QUIC 头里的方向信息: 长包头的 DCID/SCID 长度 (Chromium 客户端
   8/0, 服务器 0/8)、短包头的 DCID 都按方向不同, 它们同时也是 WARP 的部署常量
   (Retry、20 B CID)。所以 sanitized 版 = "字节通道无长度、IP/UDP 头无方向",
   不是"字节通道无方向"。剩余方向泄漏的量由 diagnose_sanitize.py 量化。

自检 (python pipeline/pcap_split/sanitize_bytes.py [--fold_dir ...]):
  S1 变换后所有包的头区逐字节相同 (上/下行, 两类)
  S2 长包头的 Length 字段全为 0
  S3 同一包类型内, 填充区 [keep_end,240) 的零字节比例与 L 不相关 (变换前强相关)
  S4 非载荷位置不再出现补零: L<240 的包, [L,240) 区间不全为 0
  S5 不改 signed_sizes / intervals / label
"""

import argparse
import copy
import glob
import hashlib
import json
import os
import random

HEADER_BYTES = 80
PAYLOAD_BYTES = 240
PACKET_BYTES = HEADER_BYTES + PAYLOAD_BYTES

HEADER_TEMPLATE = [0] * HEADER_BYTES
HEADER_TEMPLATE[0] = 0x45     # IPv4, IHL=5 (v1 全部包都是这个值)
HEADER_TEMPLATE[6] = 0x40     # DF
HEADER_TEMPLATE[9] = 17       # UDP

QUIC_V1 = (0, 0, 0, 1)
QUIC_V2 = (0x6B, 0x33, 0x43, 0xCF)
# 长包头类型位 (byte0 & 0x30) >> 4 -> 名称
LONG_TYPES = {
    QUIC_V1: {0: "initial", 1: "0rtt", 2: "handshake", 3: "retry"},
    QUIC_V2: {1: "initial", 2: "0rtt", 3: "handshake", 0: "retry"},
}


def _flow_rng(item, salt):
    key = f"{os.path.basename(item['pcap_file'])}|{salt}".encode()
    return random.Random(int(hashlib.md5(key).hexdigest()[:8], 16))


def _varint_len(b0):
    return 1 << (b0 >> 6)


def _varint_value(buf, pos):
    n = _varint_len(buf[pos])
    v = buf[pos] & 0x3F
    for i in range(1, n):
        v = (v << 8) | buf[pos + i]
    return v, n


def parse_long_header(pl, L):
    """解析载荷区开头的 QUIC 长包头。

    返回 dict(kind, keep_end, length_span):
      keep_end    : 保留真实字节到此位置 (不含), 之后填随机
      length_span : (start, end) Length 字段位置, 无则 None
    pl 是 240 B 的载荷区; L 是真实 UDP 载荷长度 (<= 240 的部分才是真实字节)。
    """
    avail = min(L, PAYLOAD_BYTES)
    if avail < 7:
        return dict(kind="short_long", keep_end=avail, length_span=None)
    b0 = pl[0]
    version = tuple(pl[1:5])
    if version == (0, 0, 0, 0):
        kind = "vn"
    elif version in LONG_TYPES:
        kind = LONG_TYPES[version][(b0 & 0x30) >> 4]
    else:
        kind = "unknown_version"
    pos = 5
    dcil = pl[pos]; pos += 1 + dcil
    if pos >= avail:
        return dict(kind=kind, keep_end=avail, length_span=None)
    scil = pl[pos]; pos += 1 + scil
    pos = min(pos, avail)
    if kind not in ("initial", "0rtt", "handshake"):
        return dict(kind=kind, keep_end=pos, length_span=None)
    if kind == "initial":
        if pos >= avail:
            return dict(kind=kind, keep_end=avail, length_span=None)
        tlen, n = _varint_value(pl, pos)
        pos = min(pos + n + tlen, avail)
    if pos >= avail:
        return dict(kind=kind, keep_end=avail, length_span=None)
    n = _varint_len(pl[pos])
    end = min(pos + n, avail)
    return dict(kind=kind, keep_end=end, length_span=(pos, end))


def sanitize_packet(pkt, rng):
    """pkt: 320 个 int。返回 (新 pkt, 解析信息)。"""
    hdr, pl = pkt[:HEADER_BYTES], pkt[HEADER_BYTES:]
    totlen = (hdr[2] << 8) | hdr[3]
    L = max(totlen - 28, 0)

    is_long = L >= 1 and (pl[0] & 0x80)
    if is_long:
        info = parse_long_header(pl, L)
    else:
        info = dict(kind="short" if L >= 1 and (pl[0] & 0x40) else "nonquic",
                    keep_end=min(L, PAYLOAD_BYTES), length_span=None)

    new_pl = list(pl[:info["keep_end"]])
    if info["length_span"]:
        s, e = info["length_span"]
        for i in range(s, e):
            new_pl[i] = 0
    new_pl += [rng.randrange(256) for _ in range(PAYLOAD_BYTES - len(new_pl))]
    info["L"] = L
    return list(HEADER_TEMPLATE) + new_pl, info


def tf_sanitize_bytes(item):
    new = copy.deepcopy(item)
    rng = _flow_rng(item, "sanitize")
    out = []
    for d in item["data"]:
        pkt = [int(x) for x in d.split()]
        assert len(pkt) == PACKET_BYTES, f"包长度 {len(pkt)} != {PACKET_BYTES}"
        s, _ = sanitize_packet(pkt, rng)
        out.append(" ".join(str(x) for x in s))
    new["data"] = out
    return new


# --------------------------------------------------------------------------- #
# [2026-10-12 ICOIN r2 #1 选项 C] 握手头归一化: 对齐 WARP 的部署常量
#
# 在 sanitize 之上, 把 QUIC 长包头改写成统一模板:
#   DCID 长度 8 + 8 B 伪随机 CID, SCID 长度 0 (Chromium 客户端 Initial 的形态),
#   Initial 的 token 长度 0 (去掉 Retry token 与 NEW_TOKEN 恢复 token, 两类都去),
#   Length 字段 2 B 全 0, 之后全部伪随机。
#   version 保留 (v1 全部为 1)。包类型位: hdrnorm 保留 (Initial / 0-RTT / Handshake
#   是协议行为, 不是部署常量); hdrnorm_types 也统一成 Initial。
# 配合 C3 (已从 pcap 层删掉 Retry 帧的 e3_folds/c123) 使用, 表示"部署常量已对齐"。
# 短包头 / 非 QUIC / VN 与 sanitize 相同处理 (v1 前 5 包里没有短包头)。
# --------------------------------------------------------------------------- #

HDRNORM_DCIL = 8


def sanitize_hdrnorm_packet(pkt, rng, norm_types):
    hdr, pl = pkt[:HEADER_BYTES], pkt[HEADER_BYTES:]
    totlen = (hdr[2] << 8) | hdr[3]
    L = max(totlen - 28, 0)
    if not (L >= 7 and (pl[0] & 0x80) and tuple(pl[1:5]) in LONG_TYPES):
        return sanitize_packet(pkt, rng)
    version = tuple(pl[1:5])
    kind = LONG_TYPES[version][(pl[0] & 0x30) >> 4]
    b0 = pl[0]
    if norm_types or kind == "retry":
        # 统一成 Initial 的类型位 (retry 不应出现在 C3 数据里, 出现则同样改写)
        init_bits = next(t for t, k in LONG_TYPES[version].items() if k == "initial")
        b0 = (b0 & 0xCF) | (init_bits << 4)
        kind = "initial"
    new_pl = [b0] + list(version) + [HDRNORM_DCIL] + [rng.randrange(256) for _ in range(HDRNORM_DCIL)] + [0]
    if kind == "initial":
        new_pl += [0]            # token length = 0
    new_pl += [0, 0]             # Length 字段 (2 B) 清零
    new_pl += [rng.randrange(256) for _ in range(PAYLOAD_BYTES - len(new_pl))]
    return list(HEADER_TEMPLATE) + new_pl, {"kind": kind, "L": L, "keep_end": len(new_pl), "length_span": None}


def _make_tf_hdrnorm(norm_types):
    def tf(item):
        new = copy.deepcopy(item)
        rng = _flow_rng(item, "hdrnorm_types" if norm_types else "hdrnorm")
        out = []
        for d in item["data"]:
            pkt = [int(x) for x in d.split()]
            assert len(pkt) == PACKET_BYTES
            s, _ = sanitize_hdrnorm_packet(pkt, rng, norm_types)
            out.append(" ".join(str(x) for x in s))
        new["data"] = out
        return new
    return tf


tf_sanitize_hdrnorm = _make_tf_hdrnorm(False)
tf_sanitize_hdrnorm_types = _make_tf_hdrnorm(True)


def self_check_hdrnorm(items, norm_types):
    """H1 头区唯一; H2 长包头 DCIL=8 / SCIL=0 / Initial token 长 0 / Length 为 0;
    H3 (norm_types) 类型位全部相同; H4 sizes/intervals/label 不变。返回 (报告, 通过与否)。"""
    from collections import Counter
    tf = tf_sanitize_hdrnorm_types if norm_types else tf_sanitize_hdrnorm
    headers, kinds, bad = set(), Counter(), 0
    for it in items:
        new = tf(it)
        for k in ("signed_sizes", "intervals", "label"):
            assert new[k] == it[k]
        for d in new["data"]:
            p = [int(x) for x in d.split()]
            headers.add(tuple(p[:HEADER_BYTES]))
            pl = p[HEADER_BYTES:]
            if pl[0] & 0x80 and tuple(pl[1:5]) in LONG_TYPES:
                kd = LONG_TYPES[tuple(pl[1:5])][(pl[0] & 0x30) >> 4]
                kinds[kd] += 1
                pos = 5
                ok = pl[pos] == HDRNORM_DCIL and pl[pos + 1 + HDRNORM_DCIL] == 0
                pos += 1 + HDRNORM_DCIL + 1
                if kd == "initial":
                    ok = ok and pl[pos] == 0
                    pos += 1
                ok = ok and pl[pos] == 0 and pl[pos + 1] == 0
                bad += (not ok)
    rep = {"distinct_headers": len(headers), "long_header_kinds_after": dict(kinds),
           "template_violations": bad}
    passed = len(headers) == 1 and bad == 0 and (not norm_types or set(kinds) <= {"initial"})
    rep["PASS"] = passed
    return rep, passed


# --------------------------------------------------------------------------- #
# 自检
# --------------------------------------------------------------------------- #

def _pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    return sxy / (sxx * syy) ** 0.5 if sxx and syy else 0.0


def self_check(items):
    from collections import Counter
    kinds = Counter()
    headers = set()
    Ls, zeros_before, zeros_after, kind_seq = [], [], [], []
    n_len_fields = n_len_nonzero = 0
    n_short_pkts = n_short_still_zero = 0
    n_hdr_tail_nonzero = 0
    for it in items:
        new = tf_sanitize_bytes(it)
        # S5
        for k in ("signed_sizes", "intervals", "label", "name", "pcap_file"):
            assert new[k] == it[k], f"S5 失败: {k} 被改动"
        rng = _flow_rng(it, "sanitize")
        for d_old, d_new in zip(it["data"], new["data"]):
            old = [int(x) for x in d_old.split()]
            nw = [int(x) for x in d_new.split()]
            if any(old[28:HEADER_BYTES]):
                n_hdr_tail_nonzero += 1
            _, info = sanitize_packet(old, rng)
            kinds[info["kind"]] += 1
            kind_seq.append(info["kind"])
            headers.add(tuple(nw[:HEADER_BYTES]))
            L = info["L"]
            Ls.append(L)
            # S3 只看被填充的区间 [keep_end, 240): 保留前缀里的零 (CID 长度等) 是
            # 服务器实现的属性, 与 L 同受服务器决定, 不是本变换要抹的长度泄漏。
            ke = HEADER_BYTES + info["keep_end"]
            span = PACKET_BYTES - ke
            zeros_before.append(sum(1 for b in old[ke:] if b == 0) / span if span else 0.0)
            zeros_after.append(sum(1 for b in nw[ke:] if b == 0) / span if span else 0.0)
            if info["length_span"]:
                n_len_fields += 1
                s, e = info["length_span"]
                if any(nw[HEADER_BYTES + s:HEADER_BYTES + e]):
                    n_len_nonzero += 1
            if L < PAYLOAD_BYTES:
                n_short_pkts += 1
                if not any(nw[HEADER_BYTES + L:]):
                    n_short_still_zero += 1
    r_before = _pearson(Ls, zeros_before)
    r_after = _pearson(Ls, zeros_after)
    # 包类型本身与 L 相关 (Retry 短且无 Length 字段, Initial 填充到 >=1200),
    # 而包类型是有意保留的 QUIC 头信息, 所以 S3 的判据是【同一包类型内】的相关。
    by_kind = {}
    for kd, L, zb, za in zip(kind_seq, Ls, zeros_before, zeros_after):
        by_kind.setdefault(kd, ([], [], []))
        by_kind[kd][0].append(L); by_kind[kd][1].append(zb); by_kind[kd][2].append(za)
    # 只有原本存在补零 (有包 L < 240) 的类型才有长度泄漏可抹; 其余类型 (如 v1 的
    # Handshake 包, L 全部 >= 578) 的填充区原本就是密文, r 只是噪声, 不计入判据。
    r_within = {kd: {"n": len(v[0]), "L_distinct": len(set(v[0])),
                     "had_zero_padding": any(L < PAYLOAD_BYTES for L in v[0]),
                     "r_before": _pearson(v[0], v[1]), "r_after": _pearson(v[0], v[2])}
                for kd, v in by_kind.items()}
    report = {
        "n_flows": len(items),
        "n_packets": len(Ls),
        "packet_kinds": dict(kinds),
        "orig_header_bytes_28_80_nonzero_packets": n_hdr_tail_nonzero,
        "S1_distinct_headers_after": len(headers),
        "S2_length_fields": n_len_fields,
        "S2_length_fields_nonzero_after": n_len_nonzero,
        "S3_corr_L_vs_fill_zero_frac_before": r_before,
        "S3_corr_L_vs_fill_zero_frac_after": r_after,
        "S3_within_kind": r_within,
        "S4_packets_L_lt_240": n_short_pkts,
        "S4_still_zero_tail_after": n_short_still_zero,
    }
    s3_ok = all(abs(v["r_after"]) < 0.1 for v in r_within.values()
                if v["had_zero_padding"] and v["L_distinct"] > 1)
    ok = (len(headers) == 1 and n_len_nonzero == 0 and s3_ok
          and n_short_still_zero == 0)
    report["PASS"] = ok
    return report


def main():
    root = os.environ.get("MASQUE_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold_dir", default=os.path.join(root, "data", "processed", "e3_folds", "full"),
                    help="含 fold*/data-*.json 的目录; 合并全部 test split = 全 504 条")
    ap.add_argument("--out", default=os.path.join(root, "outputs", "icoin_r2", "sanitize_selfcheck.json"))
    args = ap.parse_args()
    items = []
    for p in sorted(glob.glob(os.path.join(args.fold_dir, "fold*", "data-test.json"))):
        with open(p, encoding="utf-8") as f:
            items += json.load(f)
    rep = self_check(items)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    print(f"\n-> {args.out}")
    if not rep["PASS"]:
        raise SystemExit("自检未通过")


if __name__ == "__main__":
    main()
