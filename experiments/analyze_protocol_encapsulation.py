"""Protocol-correctness check of the MASQUE encapsulation story (Fig. 1 / Sec. II).

The submitted draft states that near-MTU inner QUIC packets are "split" into
smaller outer packets and that this is protocol-mandated.  This script checks
that claim against (i) the RFC text and (ii) what Cloudflare WARP actually puts
on the wire, using three independent pieces of evidence:

A.  Paired inner/outer capture (Day 0, 2026-09-14): the CloudflareWARP TUN
    adapter (inner, raw IP) and the WLAN adapter (outer, UDP/443 to
    162.159.198.2) were captured simultaneously.  Each inner packet is matched
    to the outer packet that follows (uplink) / precedes (downlink) it within a
    short window and whose size exceeds it by a plausible header overhead.
    If inner packets were split we would see inner packets with no 1:1 partner
    and outer packets pinned at the send cap followed by a remainder; if they
    are carried whole we see one partner per inner packet at a constant offset.

B.  Transport parameters of the outer connection (v1 raw pcaps, June 2026):
    the ClientHello inside the QUIC Initial is cleartext, so the WARP client's
    max_udp_payload_size / max_datagram_frame_size and SNI are readable.

C.  Packet-size distribution, first-50-packet window (the dataset the model
    sees) versus the full session (all UDP/443 packets of the raw pcaps).  The
    draft's Fig. 2 was computed on the window; this shows what the window hides.

RFC anchors (quoted in the report):
  RFC 9000 s12.4  "A frame MUST fit entirely within a QUIC packet and MUST NOT
                   span multiple packets."
  RFC 9221 s5     "DATAGRAM frames cannot be fragmented; therefore, application
                   protocols need to handle cases where the maximum datagram
                   size is limited by other factors."
  RFC 9484 s10.1  "...the endpoint SHOULD drop the IP packet and send an ICMP
                   Packet Too Big message to the sender of the dropped packet."
  RFC 9484 s6     Context ID 0 => "the Payload field contains a full IP packet".

Run (Windows python, stdlib only; needs tshark):
    python experiments/analyze_protocol_encapsulation.py
Outputs -> outputs/protocol_check/{summary.json, report.md}
"""
import os
import csv
import sys
import json
import bisect
import subprocess
import collections

_ROOT = os.environ.get("MASQUE_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(_ROOT, "data", "raw")
DAY0_DIR = os.path.join(RAW_DIR, "day0_2026-09-14")
FEAT_JSON = os.path.join(_ROOT, "data", "processed",
                         "quic_vs_masque_bi_feat", "data.json")
OUT_DIR = os.path.join(_ROOT, "outputs", "protocol_check")
TSHARK = os.environ.get("TSHARK", r"C:\Program Files\Wireshark\tshark.exe")

WARP_ENTRY = "162.159."          # Cloudflare WARP ingress /16 (v1 and Day 0)
DAY0_INNER_IP = "172.16.0.2"     # WARP TUN address on the capture host
DAY0_HOST_IP = "192.168.1.127"   # WLAN address on the capture host
MATCH_WINDOW_S = 0.020           # inner<->outer pairing window
OVERHEAD_RANGE = (30, 60)        # plausible outer-header + DATAGRAM overhead


def tshark_fields(pcap, display_filter, fields):
    cmd = [TSHARK, "-r", pcap, "-Y", display_filter, "-T", "fields"]
    for f in fields:
        cmd += ["-e", f]
    p = subprocess.run(cmd, capture_output=True, text=True)
    # tshark exits non-zero on truncated pcaps but has already emitted rows
    return [r for r in csv.reader(p.stdout.splitlines(), delimiter="\t")]


def is_lan(ip):
    return ip.startswith("192.168.") or ip.startswith("10.")


# ---------------------------------------------------------------- part A
def part_a():
    inner_p = os.path.join(DAY0_DIR, "day0_inner_masque.pcapng")
    outer_p = os.path.join(DAY0_DIR, "day0_outer_masque.pcapng")
    if not (os.path.exists(inner_p) and os.path.exists(outer_p)):
        print("[A] Day-0 paired capture not found, skipping", file=sys.stderr)
        return None
    inner = tshark_fields(inner_p, "ip || ipv6",
                          ["frame.time_epoch", "frame.len", "ip.proto",
                           "ip.src", "ipv6.nxt"])
    outer = tshark_fields(outer_p, f"udp.port==443 && ip.addr=={WARP_ENTRY}0.0/16",
                          ["frame.time_epoch", "udp.length", "ip.src"])

    def kind(r):
        if r[2] == "17":
            return "UDP4"
        if r[2] == "6":
            return "TCP4"
        return "IPv6" if r[4] else "other"

    inn = [(float(r[0]), int(r[1]), "up" if r[3] == DAY0_INNER_IP else "dn",
            kind(r)) for r in inner if r[0]]
    out = [(float(r[0]), int(r[1]) - 8, "up" if r[2] == DAY0_HOST_IP else "dn")
           for r in outer if r[0]]

    res = {}
    for d in ("up", "dn"):
        I = [x for x in inn if x[2] == d]
        O = sorted(x for x in out if x[2] == d)
        ot = [o[0] for o in O]
        used = [False] * len(O)
        ovh = collections.Counter()
        by_kind = collections.defaultdict(collections.Counter)
        unmatched_big = 0
        big = 0
        for t, L, _, k in I:
            # uplink: inner packet enters the TUN first, the outer copy follows;
            # downlink: outer copy arrives first, inner is injected after.
            lo, hi = (t, t + MATCH_WINDOW_S) if d == "up" else (t - MATCH_WINDOW_S, t)
            a, b = bisect.bisect_left(ot, lo), bisect.bisect_right(ot, hi)
            best = None
            for j in range(a, b):
                if used[j]:
                    continue
                o = O[j][1] - L
                if OVERHEAD_RANGE[0] <= o <= OVERHEAD_RANGE[1] and (
                        best is None or abs(o - 43) < abs(best[1] - 43)):
                    best = (j, o)
            if L >= 1200:
                big += 1
            if best:
                used[best[0]] = True
                ovh[best[1]] += 1
                by_kind[k][best[1]] += 1
            else:
                by_kind[k]["unmatched"] += 1
                if L >= 1200:
                    unmatched_big += 1
        outer_unmatched = collections.Counter(O[j][1] for j in range(len(O)) if not used[j])
        inner_sizes = collections.Counter(x[1] for x in I)
        outer_sizes = collections.Counter(x[1] for x in O)
        res[d] = {
            "inner_n": len(I), "outer_n": len(O),
            "matched": sum(ovh.values()),
            "overhead_hist": dict(sorted(ovh.items())),
            "overhead_mode": ovh.most_common(1)[0][0] if ovh else None,
            "inner_ge1200": big, "inner_ge1200_unmatched": unmatched_big,
            "inner_max": max(inner_sizes), "outer_max": max(outer_sizes),
            "inner_top": inner_sizes.most_common(6),
            "outer_top": outer_sizes.most_common(6),
            "outer_unmatched_n": sum(outer_unmatched.values()),
            "outer_unmatched_top": outer_unmatched.most_common(4),
            "by_inner_proto": {k: {"n": sum(v.values()), "ovh43": v[43],
                                   "unmatched": v["unmatched"]}
                               for k, v in by_kind.items()},
        }
    return res


# ---------------------------------------------------------------- part B
def part_b():
    tally = collections.Counter()
    sni = collections.Counter()
    for fn in sorted(os.listdir(RAW_DIR)):
        if not fn.endswith(".pcap"):
            continue
        cls = "Masque" if "Masque" in fn else "Direct"
        rows = tshark_fields(os.path.join(RAW_DIR, fn),
                             "quic && tls.handshake.type==1",
                             ["ip.dst", "tls.handshake.extensions_server_name",
                              "tls.quic.parameter.max_udp_payload_size",
                              "tls.quic.parameter.max_datagram_frame_size"])
        for r in rows:
            to_warp = r[0].startswith(WARP_ENTRY)
            tally[(cls, "->WARP" if to_warp else "->origin", r[2], r[3])] += 1
            if to_warp:
                sni[r[1]] += 1
    return {"clienthello_by_class_peer_maxudp_maxdgram":
            [{"class": k[0], "peer": k[1], "max_udp_payload_size": k[2],
              "max_datagram_frame_size": k[3], "n": v}
             for k, v in sorted(tally.items(), key=lambda kv: -kv[1])],
            "sni_to_warp": dict(sni)}


# ---------------------------------------------------------------- part C
def _stats(v):
    v = sorted(v)
    n = len(v)
    return {"n": n, "median": v[n // 2],
            "pct_lt120": round(100 * sum(1 for x in v if x < 120) / n, 1),
            "pct_ge1200": round(100 * sum(1 for x in v if x >= 1200) / n, 1),
            "max": v[-1]}


def part_c():
    res = {"window_first50": {}, "full_session": {}}
    d = json.load(open(FEAT_JSON))
    for cls, lab in (("Direct", 0), ("Masque", 1)):
        up, dn = [], []
        for x in d:
            if x["label"] != lab:
                continue
            for s in map(int, x["signed_sizes"].split()):
                (up if s > 0 else dn).append(abs(s))
        res["window_first50"][cls] = {"up": _stats(up), "dn": _stats(dn)}
    acc = collections.defaultdict(list)
    for fn in sorted(os.listdir(RAW_DIR)):
        if not fn.endswith(".pcap"):
            continue
        cls = "Masque" if "Masque" in fn else "Direct"
        rows = tshark_fields(os.path.join(RAW_DIR, fn), "udp.port==443 && ip",
                             ["ip.src", "ip.dst", "frame.len"])
        for src, dst, L in rows:
            up = is_lan(src)
            peer = dst if up else src
            if (cls == "Masque") != peer.startswith(WARP_ENTRY):
                continue      # Masque = tunnel to WARP only; Direct = origins only
            acc[(cls, "up" if up else "dn")].append(int(L))
    for (cls, dr), v in acc.items():
        res["full_session"].setdefault(cls, {})[dr] = _stats(v)
    return res


# ---------------------------------------------------------------- report
def write_report(A, B, C):
    L = []
    L.append("# Protocol check: is the inner packet split? (2026-09-14)\n")
    if A:
        L.append("## A. Day-0 paired inner/outer capture (Cloudflare WARP, MASQUE h3-only)\n")
        L.append("| dir | inner n | outer n | matched 1:1 | overhead mode (B) | share at mode | inner >=1200 B | of which unmatched | inner max | outer max |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for d in ("up", "dn"):
            r = A[d]
            m = r["overhead_mode"]
            L.append(f"| {d} | {r['inner_n']} | {r['outer_n']} | {r['matched']} "
                     f"({100*r['matched']/r['inner_n']:.1f}%) | {m} | "
                     f"{100*r['overhead_hist'][m]/r['matched']:.1f}% | {r['inner_ge1200']} | "
                     f"{r['inner_ge1200_unmatched']} | {r['inner_max']} | {r['outer_max']} |")
        L.append("\nPer inner protocol (whole IP packets of every family ride with the same offset):\n")
        L.append("| dir | inner proto | n | overhead = 43 B | unmatched |")
        L.append("|---|---|---|---|---|")
        for d in ("up", "dn"):
            for k, v in sorted(A[d]["by_inner_proto"].items()):
                L.append(f"| {d} | {k} | {v['n']} | {v['ovh43']} | {v['unmatched']} |")
        L.append("\nOuter packets with no inner partner are the outer connection's own "
                 "ACK-only packets: " +
                 ", ".join(f"{d}: n={A[d]['outer_unmatched_n']} top sizes {A[d]['outer_unmatched_top']}"
                           for d in ("up", "dn")) + "\n")
    L.append("## B. Outer-connection transport parameters (v1 raw pcaps, ClientHello in cleartext Initial)\n")
    L.append("| class | peer | max_udp_payload_size | max_datagram_frame_size | ClientHellos |")
    L.append("|---|---|---|---|---|")
    for r in B["clienthello_by_class_peer_maxudp_maxdgram"]:
        L.append(f"| {r['class']} | {r['peer']} | {r['max_udp_payload_size']} | "
                 f"{r['max_datagram_frame_size']} | {r['n']} |")
    L.append(f"\nSNI of ClientHellos to WARP: {B['sni_to_warp']}\n")
    L.append("## C. Packet-size distribution: first-50-packet window vs full session (frame.len, bytes)\n")
    L.append("| view | class | dir | n | median | <120 B | >=1200 B | max |")
    L.append("|---|---|---|---|---|---|---|---|")
    for view in ("window_first50", "full_session"):
        for cls in ("Direct", "Masque"):
            for dr in ("up", "dn"):
                s = C[view][cls][dr]
                L.append(f"| {view} | {cls} | {dr} | {s['n']} | {s['median']} | "
                         f"{s['pct_lt120']}% | {s['pct_ge1200']}% | {s['max']} |")
    return "\n".join(L) + "\n"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    A = part_a()
    B = part_b()
    C = part_c()
    json.dump({"A_day0_pairing": A, "B_transport_params": B, "C_size_dist": C},
              open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=2)
    rep = write_report(A, B, C)
    open(os.path.join(OUT_DIR, "report.md"), "w", encoding="utf-8").write(rep)
    print(rep)


if __name__ == "__main__":
    main()
