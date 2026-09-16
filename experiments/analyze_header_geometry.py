"""Header-geometry analysis of the Direct / MASQUE flows.

Purpose: answer "why is MASQUE separable" at the byte level, using only fields
that are readable on the wire without decryption, and checking every claim
against a prediction derived from the QUIC RFCs rather than fitted to the data.

Predictions under test
----------------------
P-A  (RFC 9000 s17.3 / s17.2)  A 1-RTT packet carrying a minimal ACK is
     1 (header byte) + |DCID| + |PN| + |ACK frame| + 16 (AEAD tag) bytes.
     |DCID| for one direction is the peer's chosen SCID, stated explicitly in
     the long headers of the handshake.  So a flow's minimum packet size in a
     given direction should equal |DCID| + 23 with |PN|=1 and a 5-byte ACK.

P-B  (RFC 9000 s14.1)  1200 bytes is the smallest max-datagram-size an
     endpoint may assume.  A client that never probes upward sits exactly
     there; one that uses a larger constant sits exactly at that constant.

P-C  (RFC 9000 s17.2.5)  Retry is an optional address-validation round trip.
     It is cleartext and arrives within the first few packets.

Run:  python experiments/analyze_header_geometry.py
      python experiments/analyze_header_geometry.py --dump-retry-index
          -> outputs/header_geometry/retry_index.json, the per-flow frame
             indices of Retry packets (consumed by
             pipeline/pcap_split/ablate_constants.py for ablation C3).
"""
import os
import re
import sys
import json
import struct
import argparse
import collections
import statistics

_ROOT = os.environ.get("MASQUE_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
FLOW_DIR = os.path.join(_ROOT, "data", "processed", "quic_vs_masque_bi")
OUT_DIR = os.path.join(_ROOT, "outputs", "header_geometry")

FNAME_RE = re.compile(r"_(\d+_\d+_\d+_\d+)_(\d+)_(\d+_\d+_\d+_\d+)_(\d+)\.pcap$")


def is_rfc1918(ip):
    o = [int(x) for x in ip.split(".")]
    return (o[0] == 10 or (o[0] == 172 and 16 <= o[1] <= 31)
            or (o[0] == 192 and o[1] == 168))


def read_pcap(path):
    """Yield (ts, linktype, frame) from a classic pcap, either endianness."""
    with open(path, "rb") as f:
        gh = f.read(24)
        if len(gh) < 24:
            return
        magic = gh[:4]
        if magic == b"\xd4\xc3\xb2\xa1":
            endian, nano = "<", False
        elif magic == b"\xa1\xb2\xc3\xd4":
            endian, nano = ">", False
        elif magic == b"\x4d\x3c\xb2\xa1":
            endian, nano = "<", True
        elif magic == b"\xa1\xb2\x3c\x4d":
            endian, nano = ">", True
        else:
            raise ValueError("not a classic pcap: %r" % magic)
        linktype = struct.unpack(endian + "I", gh[20:24])[0]
        while True:
            ph = f.read(16)
            if len(ph) < 16:
                return
            ts_s, ts_f, incl, _orig = struct.unpack(endian + "IIII", ph)
            data = f.read(incl)
            if len(data) < incl:
                return
            yield ts_s + (ts_f / 1e9 if nano else ts_f / 1e6), linktype, data


def parse_l3(frame, linktype):
    """Return (src, dst, ip_proto, l4_bytes) or None."""
    if linktype == 1:
        if len(frame) < 14:
            return None
        etype = struct.unpack("!H", frame[12:14])[0]
        off = 14
        while etype in (0x8100, 0x88A8):
            if len(frame) < off + 4:
                return None
            etype = struct.unpack("!H", frame[off + 2:off + 4])[0]
            off += 4
        if etype == 0x0800:
            ver = 4
        elif etype == 0x86DD:
            ver = 6
        else:
            return None
    elif linktype == 101:
        off, ver = 0, (frame[0] >> 4 if frame else 0)
    elif linktype == 113:
        if len(frame) < 16:
            return None
        etype = struct.unpack("!H", frame[14:16])[0]
        off = 16
        ver = 4 if etype == 0x0800 else (6 if etype == 0x86DD else 0)
    else:
        return None

    pkt = frame[off:]
    if ver == 4:
        if len(pkt) < 20:
            return None
        ihl = (pkt[0] & 0x0F) * 4
        total = struct.unpack("!H", pkt[2:4])[0]
        src = ".".join(str(b) for b in pkt[12:16])
        dst = ".".join(str(b) for b in pkt[16:20])
        return src, dst, pkt[9], (pkt[ihl:total] if total >= ihl else pkt[ihl:])
    if ver == 6:
        if len(pkt) < 40:
            return None
        plen = struct.unpack("!H", pkt[4:6])[0]
        return pkt[8:24].hex(), pkt[24:40].hex(), pkt[6], pkt[40:40 + plen]
    return None


def parse_quic_header(payload):
    """Fields readable without decryption. None if not QUIC-shaped."""
    if not payload:
        return None
    b0 = payload[0]
    if not (b0 & 0x80):
        return {"form": "short"}
    if len(payload) < 6:
        return {"form": "long", "trunc": True}
    version = struct.unpack("!I", payload[1:5])[0]
    i = 5
    dcil = payload[i]
    i += 1
    if i + dcil > len(payload):
        return {"form": "long", "trunc": True}
    i += dcil
    if i >= len(payload):
        return {"form": "long", "trunc": True}
    scil = payload[i]
    i += 1
    return {"form": "long", "version": version,
            "ptype": (b0 & 0x30) >> 4 if version != 0 else None,
            "dcil": dcil, "scil": scil}


def collect():
    rows = []
    for cls in ("Direct", "Masque"):
        d = os.path.join(FLOW_DIR, cls)
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".pcap"):
                continue
            m = FNAME_RE.search(fn)
            if not m:
                print("SKIP (unparseable name): %s" % fn, file=sys.stderr)
                continue
            a_ip, a_pt = m.group(1).replace("_", "."), int(m.group(2))
            b_ip, b_pt = m.group(3).replace("_", "."), int(m.group(4))
            # split_quic_vs_masque_bi.py normalises the five-tuple, so the two
            # endpoints appear in either order.  The capture host is the
            # RFC1918 side -- do NOT assume the first pair is the server.
            ap, bp = is_rfc1918(a_ip), is_rfc1918(b_ip)
            if ap and not bp:
                cli, srv, srv_pt = a_ip, b_ip, b_pt
            elif bp and not ap:
                cli, srv, srv_pt = b_ip, a_ip, a_pt
            else:
                print("SKIP (ambiguous endpoints): %s" % fn, file=sys.stderr)
                continue

            up, dn, cids = [], [], []
            forms = collections.Counter()
            for _ts, lt, frame in read_pcap(os.path.join(d, fn)):
                p = parse_l3(frame, lt)
                if p is None:
                    continue
                src, _dst, proto, body = p
                if proto != 17 or len(body) < 8:
                    continue
                ulen = struct.unpack("!H", body[4:6])[0]
                pay = body[8:ulen] if ulen >= 8 else body[8:]
                (up if src == cli else dn).append(len(pay))
                q = parse_quic_header(pay)
                if q:
                    forms[q["form"]] += 1
                    if q["form"] == "long" and not q.get("trunc"):
                        cids.append({"up": src == cli, "dcil": q["dcil"],
                                     "scil": q["scil"], "ptype": q["ptype"]})
            rows.append({"cls": cls, "file": fn, "srv": srv, "srv_port": srv_pt,
                         "up": up, "dn": dn, "cids": cids,
                         "forms": dict(forms)})
    return rows


def flow_dcid(r):
    """DCID length actually carried on each direction's short headers.

    Upstream packets carry the DCID the *server* chose, i.e. the SCID seen in
    server->client long headers, and vice versa.
    """
    srv_scid = [c["scil"] for c in r["cids"] if not c["up"]]
    cli_scid = [c["scil"] for c in r["cids"] if c["up"]]
    return (statistics.mode(srv_scid) if srv_scid else None,
            statistics.mode(cli_scid) if cli_scid else None)


def report(rows):
    res = {}
    print("=" * 74)
    print("P-A  minimum packet size vs |DCID| + 23")
    print("=" * 74)
    pa = []
    for cls in ("Direct", "Masque"):
        for dr, idx in (("up", 0), ("dn", 1)):
            buckets = collections.defaultdict(list)
            for r in rows:
                if r["cls"] != cls or not r[dr]:
                    continue
                cid = flow_dcid(r)[idx]
                if cid is not None:
                    buckets[cid].append(min(r[dr]))
            for cid in sorted(buckets):
                v = buckets[cid]
                med = statistics.median(v)
                pa.append({"cls": cls, "dir": dr, "dcid": cid,
                           "flows": len(v), "median_min": med,
                           "pred": cid + 23, "resid": med - (cid + 23)})
                print("  %-6s %-2s  |DCID|=%2d  n=%3d  median min=%5.1f  "
                      "pred=%3d  resid=%+5.1f"
                      % (cls, dr, cid, len(v), med, cid + 23, med - (cid + 23)))
    res["P_A"] = pa
    print("  max |residual| = %.1f B over %d groups"
          % (max(abs(x["resid"]) for x in pa), len(pa)))

    print()
    print("=" * 74)
    print("P-B  per-flow maximum packet size")
    print("=" * 74)
    pb = []
    for cls in ("Direct", "Masque"):
        for dr in ("up", "dn"):
            mx = [max(r[dr]) for r in rows if r["cls"] == cls and r[dr]]
            c = collections.Counter(mx)
            top, n = c.most_common(1)[0]
            pb.append({"cls": cls, "dir": dr, "flows": len(mx),
                       "distinct": len(c), "mode": top,
                       "mode_share": n / len(mx)})
            print("  %-6s %-2s  n=%3d  %d distinct maxima, mode=%d covers %.1f%%"
                  % (cls, dr, len(mx), len(c), top, 100.0 * n / len(mx)))
    res["P_B"] = pb

    print()
    print("=" * 74)
    print("P-C  Retry packets")
    print("=" * 74)
    pc = []
    for cls in ("Direct", "Masque"):
        rs = [r for r in rows if r["cls"] == cls]
        nf = sum(1 for r in rs if any(c["ptype"] == 3 for c in r["cids"]))
        npkt = sum(sum(1 for c in r["cids"] if c["ptype"] == 3) for r in rs)
        pc.append({"cls": cls, "flows": len(rs), "flows_with_retry": nf,
                   "retry_packets": npkt})
        print("  %-6s %4d Retry packets in %3d/%3d flows (%.1f%%)"
              % (cls, npkt, nf, len(rs), 100.0 * nf / len(rs)))
    res["P_C"] = pc

    print()
    print("=" * 74)
    print("Unfitted rules derived from the three predictions")
    print("=" * 74)
    rules = {
        "max_eq_1200": lambda r: max(r["up"] + r["dn"]) == 1200,
        "min_ge_42": lambda r: min(r["up"] + r["dn"]) >= 42,
        "has_retry": lambda r: any(c["ptype"] == 3 for c in r["cids"]),
        "max1200_and_min42": lambda r: (max(r["up"] + r["dn"]) == 1200
                                        and min(r["up"] + r["dn"]) >= 42),
    }
    usable = [r for r in rows if r["up"] + r["dn"]]
    rr = {}
    for name, fn in rules.items():
        tp = fp = tn = fn_ = 0
        for r in usable:
            pred = fn(r)
            if r["cls"] == "Masque":
                tp += pred
                fn_ += (not pred)
            else:
                fp += pred
                tn += (not pred)
        acc = (tp + tn) / len(usable)
        rr[name] = {"acc": acc, "tpr": tp / (tp + fn_), "fpr": fp / (fp + tn),
                    "tp": tp, "fp": fp, "tn": tn, "fn": fn_}
        print("  %-20s acc=%6.2f%%  TPR=%5.1f%%  FPR=%5.1f%%  (n=%d)"
              % (name, 100 * acc, 100 * tp / (tp + fn_),
                 100 * fp / (fp + tn), len(usable)))
    res["rules"] = rr

    print()
    print("  MASQUE-labelled flows whose peer is NOT a WARP entry (162.159/16):")
    off = [r for r in rows if r["cls"] == "Masque"
           and not r["srv"].startswith("162.159.")]
    print("    %d/%d flows -- these are direct traffic captured inside the"
          % (len(off), sum(1 for r in rows if r["cls"] == "Masque")))
    print("    MASQUE window, i.e. label noise, not detector error.")
    print("    peers: %s"
          % collections.Counter(".".join(r["srv"].split(".")[:2])
                                for r in off).most_common())
    res["masque_label_noise"] = {"n_off_tunnel": len(off)}
    return res


def dump_retry_index():
    """Per-flow frame indices of Retry packets, for ablation C3.

    The index space is the raw pcap record order, which is exactly what
    scapy.rdpcap() -- and therefore the feature extractor's signed_sizes /
    intervals arrays -- enumerate.  frame_lens is emitted alongside so the
    consumer can assert that alignment instead of trusting it.
    """
    out = {}
    n_retry_flows = 0
    for cls in ("Direct", "Masque"):
        d = os.path.join(FLOW_DIR, cls)
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".pcap"):
                continue
            retry_idx, frame_lens = [], []
            for i, (_ts, lt, frame) in enumerate(read_pcap(os.path.join(d, fn))):
                frame_lens.append(len(frame))
                p = parse_l3(frame, lt)
                if p is None:
                    continue
                _src, _dst, proto, body = p
                if proto != 17 or len(body) < 8:
                    continue
                ulen = struct.unpack("!H", body[4:6])[0]
                pay = body[8:ulen] if ulen >= 8 else body[8:]
                q = parse_quic_header(pay)
                if q and q["form"] == "long" and not q.get("trunc") \
                        and q["ptype"] == 3:
                    retry_idx.append(i)
            out[fn] = {"cls": cls, "n_frames": len(frame_lens),
                       "frame_lens": frame_lens, "retry_idx": retry_idx}
            n_retry_flows += bool(retry_idx)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "retry_index.json")
    with open(path, "w") as f:
        json.dump(out, f)
    print("retry index: %d/%d flows carry >=1 Retry packet (%d packets total)"
          % (n_retry_flows, len(out),
             sum(len(v["retry_idx"]) for v in out.values())))
    for cls in ("Direct", "Masque"):
        sub = [v for v in out.values() if v["cls"] == cls]
        pos = [i for v in sub for i in v["retry_idx"]]
        print("  %-6s %3d/%3d flows, Retry frame index: %s"
              % (cls, sum(1 for v in sub if v["retry_idx"]), len(sub),
                 collections.Counter(pos).most_common(6) or "-"))
    print("wrote %s" % path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-retry-index", action="store_true",
                    help="only write outputs/header_geometry/retry_index.json")
    args = ap.parse_args()
    if args.dump_retry_index:
        dump_retry_index()
        return

    rows = collect()
    print("flows parsed: %d (Direct %d / Masque %d)\n"
          % (len(rows),
             sum(1 for r in rows if r["cls"] == "Direct"),
             sum(1 for r in rows if r["cls"] == "Masque")))
    res = report(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "summary_header_geometry.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("\nwrote %s" % os.path.join(OUT_DIR, "summary_header_geometry.json"))


if __name__ == "__main__":
    main()
