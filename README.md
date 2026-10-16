# Detecting MASQUE Tunnels in Cloudflare WARP through Structural Redundancy

Code, derived data and per-fold predictions for a study of whether **MASQUE
proxy tunnels** (RFC 9298/9484, as deployed by Cloudflare WARP) can be told
apart from **native QUIC** traffic by traffic shape alone, which channels carry
the signal, and how far padding and shaping defences push back.

WARP's tunnel already identifies itself in plaintext: its handshake settings
differ from native QUIC's, and a Retry appears in every flow that reaches the
WARP ingress and in no other flow. High accuracy is therefore not the result.
The result is that the encrypted side channels reveal the tunnel independently.
On 504 sessions with session-level splitting, packet **length**, **direction**
and **timing** (the last only at a client-side vantage) each suffice alone
without reading any handshake field, and none is necessary. A defence that
closes fewer than all three, simulated offline, is undone by adversarial
retraining.

This repository is the artefact behind the paper. Every number in the paper's
tables can be re-aggregated from the prediction files under `outputs/` without
running any model; see [`REPRODUCE.md`](REPRODUCE.md) for the number-by-number
map and the commands to regenerate them from scratch.

## Repository layout

```
pipeline/                Data pipeline
  pcap_split/              pcap -> per-flow pcaps; feature transforms (ablations,
                           constants, obfuscation, byte sanitisation, relabelling,
                           steady-state segment)
  feat_extract/            flow pcaps -> model features (de-biased, direction-preserving;
                           client_anchor.py for windows that do not start at packet 1)
  relabel_util.py          MASQUE_RELABEL=1 switch used by the analysis scripts
  split_data.py            session-level train/valid/test split
  split_data_kfold.py      stratified 5-fold split sharing one fold assignment
  split_data_e4.py         destination-group hold-out splits
experiments/             Training runners, statistical analyses and plots
  selrule_util.py          MASQUE_SELRULE=acc_loss switch (model-selection rule)
dataset_scripts/         Fig. 3 of the paper
NetMambaPlus/            Submodule -> our fork of the NetMamba+ model code
data/
  raw/top_domains.txt      the 100-domain workload list (pcaps are not in git)
  processed/*_feat/        derived, de-biased features (the model inputs)
outputs/                 Per-fold predictions, summaries and logs (no checkpoints)
figures/                 Figure sources used by the paper
DATA_RELEASE.md          Capture methodology, feature schema, limitations
REPRODUCE.md             Paper number -> artefact -> command
```

Paths are derived from the repository root at runtime; set `MASQUE_ROOT` to
override. Code comments are partly in Chinese and in places cite internal
notes (`docs/...`) that are not part of this repository; nothing needs them
to run.

## Setup

```bash
git clone --recurse-submodules https://github.com/Yangxiamao/masque-detection.git
cd masque-detection
# if cloned without --recurse-submodules:  git submodule update --init

conda create -n MAMBA python=3.10.13
conda activate MAMBA
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121
pip install mamba_ssm==1.1.1
pip install -r requirements.txt
```

`NetMambaPlus/` points at our fork of
[wangtz19/NetMambaPlus](https://github.com/wangtz19/NetMambaPlus), branch
`masque-detection`. The fork adds binary-classification support (per-class loss
weighting, a tunable decision threshold), fixes three upstream bugs, and adds
the `--interval_encoding log10_decade` option described below. Without the new
flags its behaviour matches upstream.

The pre-trained checkpoint `fuse3_mamba.pth` is published by the NetMamba
authors on
[Google Drive](https://drive.google.com/drive/folders/1_IMfmpB34SM7nUSzzivd_7wAOnt2zwzE?usp=sharing);
place it at `NetMambaPlus/dataset/checkpoints/pre-trained checkpoints/fuse3_mamba.pth`.

Analyses that do not train a model (`analyze_*.py`, `check_peer_prefix.py`,
`run_e6_classical_baselines.py`, `run_const_rf_baseline.py`,
`rf_on_icoin_r2_folds.py`, `rf_attack_suite.py`, the re-aggregation in
`REPRODUCE.md`) need only `numpy`, `scipy`, `scikit-learn` and `matplotlib`.

## Data

Two classes were captured on the same client host, browser build, access
network and 100-domain list, in strict alternation over one campaign
(2026-06-16): WARP off (native QUIC/HTTP/3) and WARP on (the same workload
with Cloudflare WARP, client 2026.6.880.0, MASQUE transport). The pipeline
keeps UDP/443 flows with at least 50 packets, truncated to their first 50,
keyed on a canonicalised 5-tuple so both directions form one flow: **504
bidirectional flows = 504 sessions**.

**Labels follow the observed peer.** 259 of the 270 WARP-on flows reach the
WARP ingress (162.159.198.2) and no WARP-off flow does; the other 11 reach
other addresses directly and are labelled Direct, giving **Direct 245 /
MASQUE 259**. The committed feature files keep the capture-run labels
(234 / 270); the paper's labels are applied at run time, by
`pipeline/pcap_split/make_relabel_folds.py` for fold datasets and by
`MASQUE_RELABEL=1` (`pipeline/relabel_util.py`) for the analysis scripts. The
11 flows are listed in `outputs/e3_kfold/error_analysis_e3.json`
(`label_noise_flows`); `outputs/peer_prefix_check.json` records the peer check.

What is in git:

| Path | Contents | Size |
|---|---|---|
| `data/processed/quic_vs_masque_bi_feat/` | the 504 flows: 320 header bytes x 5 packets, signed sizes, inter-arrival times; IPs/ports replaced by direction sentinels | 5.6 MB |
| `data/processed/e6_head448_feat/`, `e6_steady_feat/` | packets 1-50 and 51-100 of the 448 flows with >= 100 packets (Section VI-A) | 9.5 MB |
| `data/processed/e6_steady_cli_feat/` | packets 51-100 again, with the direction sign anchored to the client (the non-443 endpoint) instead of the sender of packet 51 | 4.6 MB |
| `outputs/e3_kfold/fold_assignment.json` | the one fold assignment every k-fold condition reuses | |

Every other condition dataset (ablations, constant normalisation, padding,
shaping, jitter, byte sanitisation, relabelling) is a deterministic transform
of these files and is regenerated by the scripts in `REPRODUCE.md`. The raw
pcaps (3.7 GB) are not in git; see `DATA_RELEASE.md`.

> **TODO: insert the Zenodo DOI for the raw captures once the record is published.**

## Reproducing the headline result from scratch

```bash
conda activate MAMBA
# 1. raw pcaps -> 504 per-flow pcaps (needs data/raw/*.pcap)
python pipeline/pcap_split/split_quic_vs_masque_bi.py
# 2. per-flow pcaps -> features, de-biased while preserving direction
python pipeline/feat_extract/feat_quic_vs_masque_bi.py
# 3. 5-fold datasets on the shared fold assignment, then the paper's labels
python pipeline/split_data_kfold.py --feat_dir data/processed/quic_vs_masque_bi_feat \
       --out_root data/processed/e3_folds/full
python pipeline/pcap_split/make_relabel_folds.py --src_root data/processed/e3_folds \
       --dst_root data/processed/e3r_folds --only full
# 4. fine-tune one fresh model per fold; writes outputs/e3r_kfold_sel/full
MASQUE_SELRULE=acc_loss python experiments/run_kfold_e3.py \
       --fold_root data/processed/e3r_folds --out_root outputs/e3r_kfold --only full
```

Steps 1-2 need the raw captures; steps 3-4 run from the committed feature
files. Every training run passes

```
--num_packet 5 --num_packet_byte 320 --seq_len 20 --stride_size 4
--batch_size 16 --epochs 30 --model fuse3_mamba_classifier
--dataset_type byte_size_interval --size_key signed_sizes --no_amp
--interval_encoding log10_decade
```

`--interval_encoding log10_decade` is not optional. The upstream default maps
an interval x to (1+x)/(2+x), which compresses the sub-millisecond region where
the two classes differ into almost nothing; with it the timing-only condition
scores exactly the majority class. `experiments/verify_iat_encoding.py` checks
the encoding actually in effect.

`MASQUE_SELRULE=acc_loss` sets the model-selection rule: each fold keeps the
epoch with the highest validation accuracy, ties broken by the lowest
validation loss (`experiments/finetune_selrule.py`). Each fold validates on 41
sessions, so validation accuracy saturates and ties are common; keeping the
first epoch that reaches the maximum instead gives 97.02% on the relabelled
full model against 100.00% with the tie-break
(`outputs/selrule_diag/summary_selrule.json`, which compares the rules on both
label sets).

Session-level splitting is not optional either: flows of one TLS session share
endpoints and timing structure, and a flow-level split leaks.

## Known limitations

Stated up front because they bound what the results support.

- **Quote the interval estimates.** All headline numbers are pooled
  out-of-fold accuracies over stratified 5-fold cross-validation on all 504
  sessions with Wilson 95% intervals: 100.00% [99.24, 100.00] for the full
  feature set, 99.01% [97.70, 99.58] for timing only, majority class 51.39%.
  With the full model at 100%, no ablation has more than five discordant
  sessions, so "not significant" does not mean "equivalent". Folds share
  destination sites, so the intervals measure sampling stability, not
  generalisation to unseen sites.
- **Destination confounder: weakened, not excluded.** Every MASQUE flow
  reaches one WARP ingress address, while Direct traffic reaches 103 peers in
  35 /16 prefixes. Holding out every Cloudflare-destined Direct flow leaves all
  49 classified as Direct; holding out one Direct /16 at a time leaves all 168
  unseen-destination flows Direct. No split of this data can hold out the
  MASQUE-side destination.
- **Timing is vantage-specific.** 53% of the gaps in the 50-packet windows are
  below 100 us; with 50 us of zero-mean jitter timing-only NetMamba+ drops from
  99.01% to 88.89%, while the full model is unchanged (100.00% at 1 ms).
- **Asymmetric procedure.** The collector brings the tunnel up and down around
  each MASQUE domain with no matching operation on the Direct side. The
  steady-state control (packets 51-100: 99.33% against 100.00% on the first
  window of the same 448 sessions) bounds this but does not remove it.
- **Class-dependent attrition.** The >= 50-packet filter keeps 37.6% of WARP-off
  and 56.0% of WARP-on conversations, so the Direct flows are the long tail of
  their class.
- **One implementation, one vantage point, one browser, one campaign.** No
  claim is made about other MASQUE deployments, networks or times. Defences
  are simulated as post-processing of the extracted features, not implemented
  in a client.

## Earlier version

The first release of this repository (commit `1ed19b6`, 2026-09-16) used the
capture-run labels (234 / 270) and the original model-selection rule. Its
outputs (`outputs/e3_kfold/`, `e6_adv_kfold/`, `e4_cloudflare/`, ...) are kept
unchanged for comparison; the current paper's numbers come from the
`*_relabel*` and `*_sel` directories listed in `REPRODUCE.md`.

## Acknowledgements

Built on [NetMamba](https://github.com/wangtz19/NetMambaPlus) by Wang et al.;
the pre-trained checkpoint is theirs. This research was supported in part by
JSPS KAKENHI Grant Number 25K03113.

## License

Code in this repository: MIT, see `LICENSE`. The `NetMambaPlus/` submodule is
governed by its upstream license. Dataset licensing is described in
`DATA_RELEASE.md`.
