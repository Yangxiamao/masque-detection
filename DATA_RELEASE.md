# Dataset: MASQUE Tunnel vs. Native QUIC Traffic

Companion dataset description for
<https://github.com/Yangxiamao/masque-detection>. The derived features are in
this repository; the raw captures are published separately (DOI below).

Two classes of encrypted QUIC traffic captured under matched conditions:
**Direct** (native QUIC to public websites) and **Masque** (the same browsing
activity carried through a MASQUE proxy tunnel, RFC 9298).

---

## ⚠️ Before publishing: anonymisation

**Raw pcaps contain the capture host's IP address and full network metadata.**
Publishing them as-is discloses the client machine and its network. Decide one
of the following before uploading:

1. **Anonymise** with a prefix-preserving tool (e.g. `tcpmkpub`, `tcpanon`,
   or `tcprewrite --pnat`), then verify no original addresses survive; or
2. **Publish only the derived datasets** (`quic_vs_masque_bi_feat/`), whose
   feature extraction already de-biases addresses — see "De-biasing" below; or
3. **Publish raw pcaps deliberately**, having confirmed the exposed addresses
   are acceptable to disclose (e.g. a dedicated lab host on a network you
   control, with institutional approval).

Option 2 is sufficient to reproduce every result in the paper. Option 1 or 3 is
needed only if others must re-derive features from packets.

---

## Collection methodology

- **Date:** 2026-06-16, single continuous campaign (00:57–04:41 local time)
- **Workload:** automated visits to the 100 domains listed in
  `top_domains.txt` (`google.com`, `youtube.com`, `facebook.com`, ...), driven
  by headless Chromium under Playwright on a Windows 11 host, with `tshark`
  capturing on the client (100 ns timestamp resolution)
- **Design:** five iterations per class, captured **strictly alternating**
  Direct → Masque → Direct → …, roughly 17 minutes apart within a pair and 48
  minutes between pairs. Interleaving is deliberate: it prevents time-of-day
  network conditions from aligning with the class label.
- **Capture point:** client side, single vantage point
- **Tunnel:** Cloudflare WARP (client 2026.6.880.0, MASQUE transport), toggled
  with `warp-cli` between iterations. The tunnel is brought up before and torn
  down after each domain visit, with a 5 s idle wait. 259 of the 270 flows of
  the WARP-on runs reach the WARP ingress (162.159.198.2) and no WARP-off flow
  does; the other 11 reach other addresses directly, consistent with being
  captured while the tunnel was down between domains (see Labels).

## File inventory

### `raw/` — packet captures (3.7 GB, 10 files)

| File | Class | Size |
|---|---|---|
| `class0_Direct_iter1_20260616_005750.pcap` | Direct | 272.7 MB |
| `class0_Direct_iter2_20260616_014546.pcap` | Direct | 308.8 MB |
| `class0_Direct_iter3_20260616_023319.pcap` | Direct | 292.1 MB |
| `class0_Direct_iter4_20260616_031955.pcap` | Direct | 303.3 MB |
| `class0_Direct_iter5_20260616_040644.pcap` | Direct | 279.8 MB |
| `class1_Masque_iter1_20260616_011456.pcap` | Masque | 435.3 MB |
| `class1_Masque_iter2_20260616_020231.pcap` | Masque | 472.9 MB |
| `class1_Masque_iter3_20260616_024916.pcap` | Masque | 423.3 MB |
| `class1_Masque_iter4_20260616_033553.pcap` | Masque | 457.4 MB |
| `class1_Masque_iter5_20260616_042318.pcap` | Masque | 451.9 MB |

Filenames encode `class{0,1}_{label}_iter{N}_{YYYYMMDD}_{HHMMSS}`. Masque
captures are consistently larger, reflecting tunnelling overhead.

`top_domains.txt` — the 100-domain site list driving the browsing workload.

### `processed/quic_vs_masque_bi_feat/` — model-ready dataset (5.6 MB, in git)

| File | Contents |
|---|---|
| `data.json` | all 504 flows |
| `data-train.json` / `data-valid.json` / `data-test.json` | the original 403 / 50 / 51 session-level split (superseded by 5-fold CV; kept for provenance) |
| `metadata.json` | label map — `{"Direct": 0, "Masque": 1}` |

**504 flows total.** Each flow is one session, so the counts are also session
counts. The paper's headline numbers use stratified 5-fold cross-validation
over all 504 flows; the fold assignment is
`outputs/e3_kfold/fold_assignment.json`.

**Labels.** The `label` / `name` fields in these files are the **capture-run
labels**: Direct 234 (WARP off), Masque 270 (WARP on). The paper labels by the
observed peer instead: the 11 WARP-on flows whose peer is outside the WARP
ingress prefix (listed in `outputs/e3_kfold/error_analysis_e3.json`,
`label_noise_flows`) become Direct, giving **Direct 245 / MASQUE 259**. Their
handshakes match Direct's (1250 B Initial, 8-byte client connection ID, no
Retry; `outputs/handshake_fields/summary.json`). The files are left unchanged
and the relabelling is applied at run time, by
`pipeline/pcap_split/make_relabel_folds.py` for fold datasets and by
`MASQUE_RELABEL=1` for the analysis scripts, so results under both label sets
stay pairwise comparable.

Per-flow fields: `data` (first 5 packets x 320 bytes, hex, addresses and ports
replaced by direction sentinels), `signed_sizes` (50 Ethernet frame lengths,
positive = client->server), `intervals` (50 inter-arrival times in seconds),
`num_packet`, `label`, `name`, `pcap_file` (repo-relative path of the per-flow
pcap; its basename is the flow id used by every `preds.json`). Flow ids embed
the capture host's private LAN address (192.168.31.76) and the peer's public
address; the latter is what the destination-group analyses of Section VI key
on.

### `processed/e6_head448_feat/`, `processed/e6_steady_feat/` (9.5 MB, in git)

Packets 1-50 and 51-100 of the 448 flows that have at least 100 packets
(Direct 189 / MASQUE 259), used by the steady-state control of Section VI-A.

## Processing pipeline

1. **Flow segmentation.** Packets are grouped by a **canonicalised 5-tuple**
   (endpoints sorted) so that both directions of a conversation merge into one
   bidirectional flow, then sorted by timestamp to preserve the real
   client<->server interaction order. Only UDP/443 is kept. Flows shorter than
   50 packets are dropped; longer ones are truncated to 50, so every flow has
   identical length. The filter is not class-neutral: by capture run, 234 of
   623 WARP-off and 270 of 482 WARP-on conversations survive it.

2. **Feature extraction.** Per flow: first 5 packets × 320 header/payload bytes,
   a 50-step packet-size sequence, and inter-arrival times. Sizes are stored as
   `signed_sizes` — **positive = client→server, negative = server→client** — so
   direction is a first-class feature rather than being discarded.

3. **De-biasing.** Addresses and ports are removed *while preserving direction*,
   using `relative_zero_ip` + `zero_port`. `relative_zero_ip` determines
   direction from the true addresses first, then maps forward→`0.0.0.0` and
   backward→`0.0.0.1`. Naively zeroing all addresses (`zero_ip_port`) destroys
   direction and silently corrupts a bidirectional pipeline.

4. **Splitting.** `split_data_kfold.py` assigns sessions to five stratified
   folds; every condition reuses the same assignment so that conditions are
   paired. Flow-level splitting leaks information across sets, because
   same-session flows share endpoints and timing structure.

## Known limitations

State these alongside any result derived from this data.

- **Sample size.** 504 sessions; one misclassification is 0.2 pp. Quote the
  pooled out-of-fold accuracy with its Wilson interval, not single-fold values.
- **Provider confounder.** Every MASQUE flow reaches one WARP ingress address;
  Direct traffic reaches 103 peers in 35 /16 prefixes (peer-based labels). The
  hold-out controls of Section VI-B weaken but do not exclude a destination
  explanation, and no split of this data can hold out the MASQUE-side
  destination; a same-origin campaign is the proper control.
- **Labels.** The peer-based labels rest on an observable criterion (the
  peer), but the disconnect window that explains the 11 relabelled flows is
  inferred, not logged. Under the capture-run labels the full model scores
  98.81% [97.43, 99.45], with five of its six errors among the 11.
- **Asymmetric procedure.** The tunnel is set up and torn down around each
  MASQUE domain with no matching operation on the Direct side.
- **Class-dependent attrition.** The 50-packet filter keeps 37.6% of Direct and
  56.0% of MASQUE conversations, so Direct flows are the long tail of their class.
- **Single vantage point and single campaign.** One network location, one date.
  Nothing here supports claims about generalisation across networks, times, or
  client implementations.
- **Synthetic workload.** Automated top-100 browsing, not organic user traffic.

## Ethics

Traffic was generated by the authors against publicly reachable websites from a
machine under their control. No third-party user traffic was captured and no
human subjects were involved. Payloads are QUIC-encrypted and were not
decrypted. See the anonymisation notice at the top regarding capture-host
addresses.

## Suggested license

- **Data:** CC BY 4.0
- **Code** (this repository): MIT, see `LICENSE`

## Citation

```bibtex
@dataset{yang_masque_quic_2026,
  author    = {Yang, Pei},
  title     = {MASQUE Tunnel vs. Native QUIC Traffic Dataset},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {TODO},
  url       = {TODO}
}
```
