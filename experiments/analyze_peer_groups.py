"""[2026-08-23] E4 第一步: 按对端 IP 给 504 条流分组, 统计 Cloudflare 归属与 /16 分布。

回答什么 (NEXT_four_experiments_handoff.md §7):
  模型是否只在识别「对端是不是 Cloudflare」, 而非「是不是隧道」?
E4 的两个设计都需要先知道: Direct 侧有多少条流的对端落在 Cloudflare 自有网段,
以及 Direct 侧按 /16 分组后每组多大 (决定 leave-one-group-out 能不能做)。

⚠️ 两个解析陷阱 (handoff §2, 2026-08-18 踩过, 会静默出错):
  1. 流 pcap 文件名的五元组顺序不固定 —— split_quic_vs_masque_bi.py 规范化了五元组,
     `<A_ip>_<A_port>_<B_ip>_<B_port>` 中 A 不一定是服务端。判定: 采集主机在 RFC1918 一侧。
  2. RFC1918 的 172 段只有 172.16.0.0/12。写成 ip.startswith("172.") 会把
     172.217(Google) 和 172.64(Cloudflare) 误当私网 —— 而 172.64 恰恰是本实验的主角。

PHASE_SUMMARY_20260804.md §5.1 记的「约 49 条 Direct 落在 Cloudflare」来自有 bug 的解析,
本脚本给出订正后的数字。

    python experiments/analyze_peer_groups.py            # 打印统计
    python experiments/analyze_peer_groups.py --dump     # 另写 peer_groups.json
"""

import argparse
import collections
import ipaddress
import json
import os

_ROOT = os.environ.get("MASQUE_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
FEAT_DIR = os.path.join(_ROOT, "data", "processed", "quic_vs_masque_bi_feat")
OUT_DIR = os.path.join(_ROOT, "outputs", "e4_cloudflare")

# Cloudflare 官方公布的 IPv4 anycast 段 (https://www.cloudflare.com/ips-v4/,
# 快照 2026-08-23) + 170.114.0.0/16。后者不在 anycast 列表里但属 AS13335,
# 本数据集中出现 8 条 Direct 流, handoff §7.3 已把它列为 Cloudflare 网段。
CLOUDFLARE_NETS = [ipaddress.ip_network(c) for c in [
    "103.21.244.0/22", "103.22.200.0/22", "103.31.4.0/22", "104.16.0.0/13",
    "108.162.192.0/18", "131.0.72.0/22", "141.101.64.0/18", "162.158.0.0/15",
    "172.64.0.0/13", "173.245.48.0/20", "188.114.96.0/20", "190.93.240.0/20",
    "197.234.240.0/22", "198.41.128.0/17", "170.114.0.0/16",
]]

# WARP 隧道入口。它也在 Cloudflare 段内 (162.158.0.0/15), 但与 CDN 边缘是
# 不同服务、行为不同 —— handoff §7.6 的措辞纪律要求把两者分开统计。
WARP_ENTRY = ipaddress.ip_network("162.159.0.0/16")


def is_rfc1918(ip):
    o = [int(x) for x in ip.split(".")]
    return (o[0] == 10 or (o[0] == 172 and 16 <= o[1] <= 31)
            or (o[0] == 192 and o[1] == 168))


def parse_endpoints(basename):
    """从流 pcap 文件名解析 (客户端, 对端, 对端端口)。返回 None 表示无法判定。"""
    segs = basename.replace(".pcap", "").split("_")
    quint = segs[-10:]
    a_ip, a_pt = ".".join(quint[0:4]), int(quint[4])
    b_ip, b_pt = ".".join(quint[5:9]), int(quint[9])
    ap, bp = is_rfc1918(a_ip), is_rfc1918(b_ip)
    if ap and not bp:
        return a_ip, b_ip, b_pt
    if bp and not ap:
        return b_ip, a_ip, a_pt
    return None


def in_cloudflare(ip):
    a = ipaddress.ip_address(ip)
    return any(a in n for n in CLOUDFLARE_NETS)


def slash16(ip):
    return ".".join(ip.split(".")[:2])


def load_flows(feat_dir=FEAT_DIR):
    """合并三个 split, 给每条流补上对端信息。返回按 basename 排序的列表。"""
    items = []
    for sp in ("train", "valid", "test"):
        with open(os.path.join(feat_dir, f"data-{sp}.json"), encoding="utf-8") as f:
            for it in json.load(f):
                items.append({"key": os.path.basename(it["pcap_file"]),
                              "label": it["label"], "name": it["name"],
                              "split": sp})
    rows = []
    for it in items:
        ep = parse_endpoints(it["key"])
        if ep is None:
            raise SystemExit(f"端点无法判定 (两端都是/都不是 RFC1918): {it['key']}")
        cli, peer, port = ep
        rows.append({**it, "client": cli, "peer": peer, "peer_port": port,
                     "peer16": slash16(peer), "peer24": ".".join(peer.split(".")[:3]),
                     "is_cf": in_cloudflare(peer),
                     "is_warp_entry": ipaddress.ip_address(peer) in WARP_ENTRY})
    rows.sort(key=lambda r: r["key"])
    return rows


def report(rows):
    n = len(rows)
    direct = [r for r in rows if r["name"] == "Direct"]
    masque = [r for r in rows if r["name"] == "Masque"]
    print(f"flows {n}  (Direct {len(direct)} / Masque {len(masque)})\n")

    print("=" * 74)
    print("1. Cloudflare 归属 (订正后的解析)")
    print("=" * 74)
    for tag, sub in (("Direct", direct), ("Masque", masque)):
        cf = [r for r in sub if r["is_cf"]]
        warp = [r for r in sub if r["is_warp_entry"]]
        cf_nonwarp = [r for r in cf if not r["is_warp_entry"]]
        print(f"  {tag:<7} n={len(sub):3d}   对端在 Cloudflare 段 {len(cf):3d} "
              f"({100*len(cf)/len(sub):5.1f}%)   其中 WARP 入口 162.159/16 {len(warp):3d}"
              f"   CDN 边缘(非 WARP) {len(cf_nonwarp):3d}")
    print()
    print("  ★ E4 主实验用的是 Direct 侧「对端在 Cloudflare 段且非 WARP 入口」的流:")
    d_cf = [r for r in direct if r["is_cf"] and not r["is_warp_entry"]]
    print(f"    {len(d_cf)} 条, 分布 /16: "
          f"{collections.Counter(r['peer16'] for r in d_cf).most_common()}")
    print(f"    (PHASE_SUMMARY_20260804 §5.1 记的『约 49 条』来自有 bug 的解析)")

    print()
    print("=" * 74)
    print("2. 对端 /16 分组 (E4-B leave-one-group-out 的分组依据)")
    print("=" * 74)
    for tag, sub in (("Direct", direct), ("Masque", masque)):
        c = collections.Counter(r["peer16"] for r in sub)
        print(f"  {tag} : {len(c)} 组")
        for g, k in c.most_common():
            cf = "CF" if in_cloudflare(g + ".0.0") else "  "
            print(f"      {g:<10} {cf}  {k:3d} 条")
    both = (set(r["peer16"] for r in direct) & set(r["peer16"] for r in masque))
    print(f"  两类共享的 /16 组: {sorted(both) or '(无)'}")

    print()
    print("=" * 74)
    print("3. 粒度对照")
    print("=" * 74)
    for key, name in (("peer", "full IP"), ("peer24", "/24"), ("peer16", "/16")):
        print(f"  {name:<8}: 全集 {len(set(r[key] for r in rows)):3d} 组   "
              f"Direct {len(set(r[key] for r in direct)):3d}   "
              f"Masque {len(set(r[key] for r in masque)):3d}")

    d16 = collections.Counter(r["peer16"] for r in direct)
    usable = [g for g, k in d16.items() if k >= 8]
    print()
    print(f"  Direct 侧 >= 8 条流的 /16 组: {len(usable)} 个, "
          f"覆盖 {sum(d16[g] for g in usable)}/{len(direct)} 条 Direct 流")
    print("  -> E4-B 对这些组做 leave-one-group-out (每组一次训练)")

    return {
        "n_flows": n,
        "n_direct": len(direct), "n_masque": len(masque),
        "direct_cf_total": sum(1 for r in direct if r["is_cf"]),
        "direct_cf_nonwarp": len(d_cf),
        "direct_cf_nonwarp_by16": dict(collections.Counter(r["peer16"] for r in d_cf)),
        "masque_warp_entry": sum(1 for r in masque if r["is_warp_entry"]),
        "masque_off_tunnel": sum(1 for r in masque if not r["is_warp_entry"]),
        "direct_by16": dict(d16),
        "masque_by16": dict(collections.Counter(r["peer16"] for r in masque)),
        "shared_16": sorted(both),
        "direct_groups_ge8": sorted(usable),
        "cloudflare_nets": [str(c) for c in CLOUDFLARE_NETS],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="store_true",
                    help="另写 outputs/e4_cloudflare/peer_groups.json (逐流对端表)")
    args = ap.parse_args()

    rows = load_flows()
    res = report(rows)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "summary_peer_groups.json"), "w",
              encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {os.path.join(OUT_DIR, 'summary_peer_groups.json')}")

    if args.dump:
        p = os.path.join(OUT_DIR, "peer_groups.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump({r["key"]: {k: r[k] for k in
                                  ("label", "name", "peer", "peer16", "peer24",
                                   "is_cf", "is_warp_entry")} for r in rows},
                      f, indent=2)
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
