"""Why do some inner packets of the Day-0 paired capture have no outer partner?

Follow-up to analyze_protocol_encapsulation.py (part A), prompted by the
professor's 2026-10-16 comment on Sec. II: state the denominator of the
"95--98 %" figure and explain, as far as the traces allow, the packets that
did not pair.

Re-runs the same greedy pairing (20 ms window, overhead 30--60 B, closest to
43 B wins) and then classifies every unmatched inner packet:

  edge       outside the time span covered by both captures
  late       pairs at exactly +43 B once the window is widened to 200 ms
  coalesced  part of a run of 2--8 unmatched inner packets whose sizes, plus
             per-packet framing, add up to one unmatched outer packet
  other      none of the above

Run (Windows python, stdlib only; needs tshark):
    python experiments/diagnose_encapsulation_unmatched.py
Output -> outputs/protocol_check/unmatched_diag.json (+ printed summary)
"""
import os
import sys
import json
import bisect
import collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_protocol_encapsulation import (  # noqa: E402
    tshark_fields, DAY0_DIR, OUT_DIR, WARP_ENTRY, DAY0_INNER_IP, DAY0_HOST_IP,
    MATCH_WINDOW_S, OVERHEAD_RANGE)

WIDE_WINDOW_S = 0.200
COALESCE_MAX_K = 8
COALESCE_WINDOW_S = 0.020
# outer = sum(inner) + 43 + (k-1) * per-extra-frame cost.  For k = 2 the
# residual peaks sharply at +5/+6 B (up 35/39, dn 912/996 pairs within +-20 B);
# with the 43 B base replaced by 73 or 103 B the same search finds ~3 pairs per
# 1 B bin, so the peak is not a chance fit.
EXTRA_FRAME_COST = (5, 6)
ACK_ONLY_MAX = 60


def load():
    inner = tshark_fields(os.path.join(DAY0_DIR, "day0_inner_masque.pcapng"),
                          "ip || ipv6",
                          ["frame.time_epoch", "frame.len", "ip.proto",
                           "ip.src", "ipv6.nxt", "ip.dst"])
    outer = tshark_fields(os.path.join(DAY0_DIR, "day0_outer_masque.pcapng"),
                          f"udp.port==443 && ip.addr=={WARP_ENTRY}0.0/16",
                          ["frame.time_epoch", "udp.length", "ip.src"])
    inn = []
    for r in inner:
        if not r[0]:
            continue
        d = "up" if r[3] == DAY0_INNER_IP else "dn"
        peer = (r[5] if d == "up" else r[3]) or "v6"
        inn.append((float(r[0]), int(r[1]), d, peer))
    out = [(float(r[0]), int(r[1]) - 8, "up" if r[2] == DAY0_HOST_IP else "dn")
           for r in outer if r[0]]
    return inn, out


def pair(I, O, d, window):
    ot = [o[0] for o in O]
    used = [False] * len(O)
    match = [None] * len(I)
    for i, (t, L, _, _) in enumerate(I):
        lo, hi = (t, t + window) if d == "up" else (t - window, t)
        a, b = bisect.bisect_left(ot, lo), bisect.bisect_right(ot, hi)
        best = None
        for j in range(a, b):
            if used[j]:
                continue
            o = O[j][1] - L
            if OVERHEAD_RANGE[0] <= o <= OVERHEAD_RANGE[1] and (
                    best is None or abs(o - 43) < abs(best[1] - 43)):
                best = (j, o)
        if best:
            used[best[0]] = True
            match[i] = best
    return match, used


def main():
    inn, out = load()
    t0 = max(min(x[0] for x in inn), min(x[0] for x in out))
    t1 = min(max(x[0] for x in inn), max(x[0] for x in out))
    res = {"overlap_s": round(t1 - t0, 3)}
    for d in ("up", "dn"):
        I = sorted(x for x in inn if x[2] == d)
        O = sorted(x for x in out if x[2] == d)
        match, used = pair(I, O, d, MATCH_WINDOW_S)
        un = [i for i in range(len(I)) if match[i] is None]
        cls = {}
        # edge
        for i in un:
            if not (t0 + MATCH_WINDOW_S <= I[i][0] <= t1 - MATCH_WINDOW_S):
                cls[i] = "edge"
        # coalesced: runs of consecutive unmatched inner packets vs one
        # unmatched non-ACK outer packet
        ot = [o[0] for o in O]
        free_outer = [j for j in range(len(O)) if not used[j] and O[j][1] > ACK_ONLY_MAX]
        un_set = [i for i in un if i not in cls]
        un_t = [I[i][0] for i in un_set]
        taken = set()
        n_outer_coalesced = 0
        k_hist = collections.Counter()
        for j in free_outer:
            to, Lo = O[j][0], O[j][1]
            lo, hi = (to - COALESCE_WINDOW_S, to) if d == "up" else (to, to + COALESCE_WINDOW_S)
            a, b = bisect.bisect_left(un_t, lo), bisect.bisect_right(un_t, hi)
            cand = [un_set[p] for p in range(a, b) if un_set[p] not in taken]
            hit = None
            # contiguous runs only (one outer packet bundles packets that
            # arrived back to back)
            for s in range(len(cand)):
                tot = 0
                for e in range(s, min(len(cand), s + COALESCE_MAX_K)):
                    tot += I[cand[e]][1]
                    k = e - s + 1
                    if k < 2:
                        continue
                    extra = Lo - tot - 43
                    if EXTRA_FRAME_COST[0] * (k - 1) <= extra <= EXTRA_FRAME_COST[1] * (k - 1):
                        hit = cand[s:e + 1]
                        break
                    if tot + 43 > Lo:
                        break
                if hit:
                    break
            if hit:
                n_outer_coalesced += 1
                k_hist[len(hit)] += 1
                for i in hit:
                    taken.add(i)
                    cls[i] = "coalesced"
        # late: widen the window for what is left
        rest = [i for i in un if i not in cls]
        match_w, _ = pair(I, O, d, WIDE_WINDOW_S)
        for i in rest:
            if match_w[i] is not None and match_w[i][1] == 43:
                cls[i] = "late"
        for i in un:
            cls.setdefault(i, "other")
        c = collections.Counter(cls.values())
        other = [i for i in un if cls[i] == "other"]
        sz = collections.Counter(I[i][1] for i in other)
        peers_un = collections.Counter(I[i][3] for i in un)
        peers_all = collections.Counter(x[3] for x in I)
        res[d] = {
            "inner_n": len(I), "outer_n": len(O),
            "matched": sum(1 for m in match if m),
            "matched_ovh43": sum(1 for m in match if m and m[1] == 43),
            "unmatched": len(un),
            "unmatched_by_reason": dict(c),
            "outer_packets_holding_coalesced_inner": n_outer_coalesced,
            "coalesce_k_hist": dict(sorted(k_hist.items())),
            "other_top_sizes": sz.most_common(8),
            "unmatched_top_peers": [(p, n, peers_all[p]) for p, n in peers_un.most_common(5)],
            "outer_unmatched_ack_only": sum(1 for j in range(len(O))
                                            if not used[j] and O[j][1] <= ACK_ONLY_MAX),
            "outer_unmatched_non_ack": sum(1 for j in range(len(O))
                                           if not used[j] and O[j][1] > ACK_ONLY_MAX),
        }
    json.dump(res, open(os.path.join(OUT_DIR, "unmatched_diag.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
