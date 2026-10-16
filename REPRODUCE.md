# Reproducing the paper's numbers

Two levels of reproduction are supported.

1. **Re-aggregate** (seconds, no GPU): every accuracy in the paper's tables is
   a pooled out-of-fold count over the per-sample prediction files committed
   under `outputs/`. The snippet below recomputes them.
2. **Regenerate** (GPU, hours): rebuild the condition datasets from the
   committed features and fine-tune a fresh model per fold. Conditions that
   need the raw pcaps are marked.

All commands run from the repository root inside the `MAMBA` environment.

The paper uses the **peer-based labels** (Direct 245 / MASQUE 259, see
`README.md`) and the **acc_loss model-selection rule**. In this repository
that means:

- NetMamba+ results live in directories ending in `_sel`
  (`MASQUE_SELRULE=acc_loss`), trained on the relabelled fold datasets
  (`e3r_*`, `e6r_*`, `--labels relabel`, `--relabel`);
- classical models and analyses live in directories ending in `_relabel`
  (`MASQUE_RELABEL=1`); they have no epoch to select, so they have no `_sel`
  variant.

Directories without either suffix (`outputs/e3_kfold/`, `e6_adv_kfold/`,
`e4_cloudflare/`, ...) are the first release (capture-run labels, original
selection rule), kept for comparison and not used by the current paper.

## 1. Re-aggregating from the committed predictions

Each `preds.json` maps a flow id to `{"pred", "label", "prob1"}` for the flows
in that fold's test set; concatenating the five folds gives every session
exactly once, and the labels in it are already the peer-based ones.

```python
import json, glob
def oof(pattern):
    rows = [v for p in glob.glob(pattern) for v in json.load(open(p)).values()]
    return len(rows), round(100 * sum(r["pred"] == r["label"] for r in rows) / len(rows), 2)

print(oof("outputs/e3r_kfold_sel/full/fold*/preds.json"))                     # (504, 100.0)
print(oof("outputs/e3r_kfold_sel/nosizes_nobytes/fold*/preds.json"))          # (504, 99.01)
print(oof("outputs/e6r_adv_kfold_sel/preds/clean/t3_padshape/fold*.json"))    # (504, 56.35)
print(oof("outputs/icoin_r2/relabel_sel/t3dir_raw/fold*/preds.json"))         # (504, 73.61)
print(oof("outputs/e3_kfold_sel/steady_cli_suff_dir/fold*/preds.json"))       # (448, 89.96)
```

Paired tests, Wilson intervals and Holm correction are in
`outputs/relabel_sel/mcnemar_table1_family11.json` (Table I family),
`outputs/relabel_sel/summary_relabel_sel.json` (Table II family, jitter, E4)
and the per-experiment summaries below; `python experiments/summarize_relabel_sel.py`
regenerates the first two from the prediction files.

## 2. Number-by-number map

`cond` refers to `outputs/e3r_kfold_sel/<cond>/fold{0..4}/preds.json` unless
another path is given. Section numbers follow the paper.

### Section II — Background (paired capture)

| Paper statement | Artefact | Regenerate |
|---|---|---|
| 85.4% of 886 uplink / 72.5% of 8335 downlink inner packets pair one-to-one; 98.3% / 95.0% of pairs differ by 43 B; tunnel MTU 1300 B in the paired capture | `outputs/protocol_check/summary.json`, `report.md` | `python experiments/analyze_protocol_encapsulation.py` (Day-0 dual-interface capture, not in git) |
| Most unpaired packets bundled two or three per outer packet (87 of 129 up, 1935 of 2295 down) | `outputs/protocol_check/unmatched_diag.json` | `python experiments/diagnose_encapsulation_unmatched.py` (same capture) |
| Outer ACK-only packets 42-44 B in the dataset | `signed_sizes` of `data/processed/quic_vs_masque_bi_feat/data.json` (frame length = UDP payload + 42 B) | direct count |

### Section III — Dataset and pipeline

| Paper statement | Artefact | Regenerate |
|---|---|---|
| 504 sessions; 245 Direct / 259 MASQUE by observed peer; 259 of 270 WARP-on flows reach 162.159, no WARP-off flow does | `outputs/peer_prefix_check.json`; `outputs/e3_kfold/error_analysis_e3.json` (`label_noise_flows`) | `python experiments/check_peer_prefix.py` |
| The 11 relabelled flows have Direct's handshake (1250 B Initial = 1292 B frame, 8-byte client Initial DCID, no Retry) | `outputs/handshake_fields/summary.json` | `python experiments/analyze_handshake_fields.py` |
| Capture-run labels: 98.81% [97.43, 99.45], five of six errors among the 11 | `outputs/selrule_diag/orig/fold*/preds_acc_loss.json`, `summary_selrule.json` (`labels.orig.rules.acc_loss`) | `python experiments/run_selrule_diag.py` |
| Attrition 56.0% vs 37.6% of conversations, Fisher p = 1.1e-9, medians 2019 vs 33 packets (by capture run) | `outputs/e6_attrition/summary_attrition.json` | `python experiments/analyze_flow_attrition.py` (raw pcaps) |
| Model selection: highest validation accuracy, ties by lowest validation loss | `outputs/selrule_diag/summary_selrule.json` (relabelled full: first-max 97.02% vs tie-break 100.00%) | `python experiments/run_selrule_diag.py` |
| Fig. 3; 67% of Direct downlink >= 1200 B (median 1242), 54% of MASQUE downlink < 120 B (median 116) | `figures/conf_size_distribution_relabel.png`, `figures/short_size_distribution_relabel_stats.json` | `python dataset_scripts/plot_short_size_relabel.py`; `python dataset_scripts/plot_conf_size_relabel.py` |
| Whole sessions: 77.6% of WARP-on vs 90.4% of WARP-off downlink packets >= 1200 B | `outputs/protocol_check/summary.json` (`pct_ge1200`) | `python experiments/analyze_protocol_encapsulation.py` (raw pcaps) |
| Raw 40-dim vector: logistic regression 99.80 +/- 0.40, random forest 100 in every fold | `outputs/e6_classical_relabel/summary_classical_baselines.json` (`raw_first20_*`) | `MASQUE_RELABEL=1 python experiments/run_e6_classical_baselines.py` |

### Section IV — Feature redundancy (Table I)

| Row | NetMamba+ | RF column |
|---|---|---|
| Full baseline 100.00, CM [[245,0],[0,259]], ROC-AUC 1.0000 | `full`; `outputs/e6_operating_points_relabel_sel/operating_points_full.json` | `outputs/icoin_r2/relabel/rf_table1.json` |
| No-Intervals / No-Direction / No-Bytes 100.00 | `no_intervals`, `no_direction`, `no_bytes` | same |
| No-Length 99.60 (strict) | `outputs/icoin_r2/relabel_sel/nolen_pad_nobytes/fold*/preds.json` | `outputs/icoin_r2/relabel/rf_summary.json` (`nolen_pad_nobytes`) |
| No-Sizes 99.21 | `no_sizes` | `rf_table1.json` |
| Length / Direction / Timing / Bytes only 100.00 / 99.60 / 99.01 / 75.99 | `suff_mag`, `suff_dir`, `nosizes_nobytes`, `suff_bytes` | `rf_table1.json` |
| p column (exact McNemar, Holm over 11) | `outputs/relabel_sel/mcnemar_table1_family11.json` | |

IV-B: random forest on 24 length statistics 100% in every fold ->
`outputs/e6_classical_relabel/summary_classical_baselines.json`
(`len24_first20_RandomForest`).

IV-C (observer jitter):

| Statement | Artefact |
|---|---|
| 53% of the gaps in the 50-packet windows are below 100 us | `outputs/window_gaps_relabel/summary.json` (`window_50.all`: 13098 / 24696) |
| Sub-100 us RF importance 38.3% -> 1.8% at 50 us; RF 83.1-89.5% across sigma | `outputs/e6_jitter_relabel/summary_e6_jitter_stats.json` |
| Timing-only 99.01 -> 88.89 at 50 us (p = 4e-13), 67.86 at 5 ms | `jit50us`, `jit200us`, `jit1ms`, `jit5ms`; p in `summary_relabel_sel.json` (`jitter`) |
| Full model 100.00 at 1 ms | `jit1ms_full` |

### Section V — Attack and defence (Tables II and III)

Table II, all 24 cells:
`outputs/e6r_adv_kfold_sel/preds/<detector>/<transform>/fold{0..4}.json`,
detectors `clean`, `advpad`, `advshape`, `advmo` (rows), transforms
`t1_clean`, `t2_pad`, `t2_strict`, `t3_padshape`, `t3_absdir`, `t2_pad_mo`
(columns), plus `t_shape` (shaping alone, quoted in V-C). Summary and
confusion matrices: `outputs/e6r_adv_kfold_sel/summary_e6_adv_kfold.json`;
Holm over 21 comparisons: `outputs/relabel_sel/summary_relabel_sel.json`
(`table2`); the padding-trained detector on T3 against the pad+shape-trained
one (10 sessions, p = 0.002 uncorrected):
`summary_e6_adv_kfold.json` (`transfer_advpad_vs_advshape_on_t3`).

Table III (each row retrained with train = test transform):

| Row | NetMamba+ | RF |
|---|---|---|
| a T3-dir | `outputs/icoin_r2/relabel_sel/t3dir_raw/` | `outputs/icoin_r2/relabel/rf_summary.json` (same keys) |
| b + header template, payload randomised | `t3dir_san` | |
| c + Retry removed | `t3dir_c123_san` | |
| d + CID lengths and token set to Chromium's | `t3dir_c123_hdr` | |
| e + long-header packet types unified | `t3dir_c123_hdrtype` | |
| f all bytes zeroed | `t3dir_strict` | |

Summary with Wilson intervals and the McNemar tests against the majority
predictor: `outputs/icoin_r2/relabel_sel/summary.json`. Clean-trained forest
on the Table II columns (>= 99.8% wherever the byte channel is kept):
`outputs/icoin_r2/relabel/rf_table2.json`. What the byte channel still leaks
after T3-dir, and the self-checks of the header/payload rewriting:
`outputs/icoin_r2/sanitize_leak_diag.json`, `sanitize_selfcheck.json`,
`hdrnorm_selfcheck_c123.json` (`python experiments/diagnose_sanitize.py`;
`python pipeline/pcap_split/sanitize_bytes.py`).

V-D: padding to 1350 B inflates the 50-packet windows 3.73x for MASQUE (mean
packet 361 B) and 1.77x for Direct ->
`outputs/e6_defense_cost_relabel/defense_cost.json`
(`MASQUE_RELABEL=1 python experiments/analyze_defense_cost.py`). The
MASQUE-only detector (100.00 on its own mixture, 53.57 with both classes
padded) is the `advmo` row of Table II.

### Section VI — Alternative explanations

| Statement | Artefact | Regenerate |
|---|---|---|
| Steady state (packets 51-100) 99.33% [98.05, 99.77] vs 100.00% on the first window of the same 448 sessions, p = 0.25, majority 57.81% | `outputs/e3_kfold_sel/steady/`, `steady_cli/`, `head448/`; `outputs/e6_steady_sel/summary_e6_steady.json`, `summary_steady_client_anchor.json` | `MASQUE_SELRULE=acc_loss python experiments/run_e6_steady.py --stage train` |
| Steady-state single channels: length 99.78, direction 89.96, timing 96.65 (RF 99.33 / 91.74 / 97.77); direction loses 44 and gains 2 against the full model, timing loses 12 | `outputs/e3_kfold_sel/steady_suff_len/`, `steady_cli_suff_dir/`, `steady_suff_time/`; `outputs/e6_steady_sel/summary_steady_suff.json`, `summary_steady_client_anchor.json` | after the row above: `python experiments/run_steady_suff.py --stage all`, then `python experiments/run_steady_client_anchor.py --stage all` (re-uses the committed `e6_steady_cli_feat`; extraction runs only if it is missing) |
| Direct 103 peers in 35 /16; all 259 MASQUE flows to 162.159.198.2 | `outputs/peer_prefix_check.json` (`relabelled`) | `python experiments/check_peer_prefix.py` |
| All 49 Cloudflare-destined Direct flows held out stay Direct, CI [92.7, 100.0]; leave-one-/16-out 168/168, CI [97.8, 100.0] | `outputs/e4_cloudflare_relabel_sel/cf_holdout/preds.json`, `groupout_*/preds.json`, `summary_e4.json` | `MASQUE_RELABEL=1 python pipeline/split_data_e4.py --all`; `MASQUE_SELRULE=acc_loss MASQUE_RELABEL=1 python experiments/run_e4_cloudflare.py` |
| P-A/P-B/P-C: 1250 B Initial in all 245 Direct, 1200 B in 97% of MASQUE; Retry in 259/259 vs 0/245 | `outputs/header_geometry_relabel/summary_header_geometry.json` | `MASQUE_RELABEL=1 python experiments/analyze_header_geometry.py` (per-flow pcaps) |
| Zero-training rules 98.02% (max packet = 1200) and 100.00% (contains Retry); max-packet rule TPR 0.0% at FPR 0.0% after C123; RF on 24 length statistics 100% before and after | `outputs/const_rf_baseline_relabel/summary_const_rf_baseline.json` (parts B and C) | `MASQUE_RELABEL=1 python experiments/run_const_rf_baseline.py` |
| C1 99.80, C2 / C3 / C123 / C123 without bytes 100.00 | `c1`, `c2`, `c3`, `c123`, `c123_nobytes`; Holm in `mcnemar_table1_family11.json` | `python pipeline/pcap_split/ablate_constants.py` (per-flow pcaps; C3 re-extracts after removing Retry) |

The 448 steady-state sessions contain none of the 11 relabelled flows (all
have fewer than 100 packets), so the steady-state datasets are identical under
both label sets and their scripts take no relabel switch. The
`e6_steady_feat` window anchors the direction sign to the sender of packet 51;
`e6_steady_cli_feat` and `pipeline/feat_extract/client_anchor.py` anchor it to
the client, as the paper states. Length-only and timing-only inputs are
byte-identical under both anchors (checked by `run_steady_client_anchor.py
--stage verify`), so only the full and direction-only models were retrained.

### Section VII — Limitations

Precision at 1% / 0.1% prevalence with a 1% FPR (50.3% / 9.1%), smallest
expressible FPR 1/245 = 0.41%:
`outputs/e6_operating_points_relabel_sel/operating_points_full.json`
(`MASQUE_RELABEL=1 MASQUE_SELRULE=acc_loss python experiments/analyze_e6_operating_points.py --condition full`).

## 3. Regenerating the relabelled runs

1. **Condition datasets** (original labels). Each condition id in `CONDITIONS`
   of `experiments/run_kfold_e3.py` names a dataset under `data/processed/`,
   produced from the committed features by
   `pipeline/pcap_split/simulate_obfuscation_bi.py` (`--ablation`, `--all`,
   `--e6`) or `pipeline/pcap_split/ablate_constants.py` (C1-C3, per-flow
   pcaps). `run_kfold_e3.py` splits each into `data/processed/e3_folds/<cond>`
   on the shared fold assignment the first time it is run; to build the folds
   without training, call `pipeline/split_data_kfold.py --feat_dir
   data/processed/<dataset> --out_root data/processed/e3_folds/<cond>`.
   `pipeline/pcap_split/verify_e6_datasets.py` asserts the transforms did what
   they claim.
2. **Relabel** the folds; only the 11 labels change, so samples, folds and
   transforms stay byte-identical and the two label sets can be compared
   pairwise:
   ```bash
   python pipeline/pcap_split/make_relabel_folds.py \
          --src_root data/processed/e3_folds --dst_root data/processed/e3r_folds
   MASQUE_RELABEL=1 python pipeline/split_data_e4.py --all
   ```
3. **NetMamba+** (GPU): `bash experiments/queue_relabel_sel.sh` runs, with
   `MASQUE_SELRULE=acc_loss`, Table I, Table II (`run_e6_adv_kfold.py
   --relabel`), Table III and No-Length (`run_icoin_r2.py --labels relabel`),
   the constants, the jitter series, E4 and the operating points. Then
   `python experiments/summarize_relabel_sel.py`.
4. **Random forests and analyses** (CPU):
   ```bash
   python experiments/rf_on_icoin_r2_folds.py --labels relabel      # Table III RF, No-Length RF
   python experiments/rf_attack_suite.py --labels relabel --part table1
   python experiments/rf_attack_suite.py --labels relabel --part table2
   export MASQUE_RELABEL=1
   python experiments/run_e6_classical_baselines.py
   python experiments/run_e6_jitter_stats.py
   python experiments/run_const_rf_baseline.py
   python experiments/analyze_defense_cost.py
   python experiments/analyze_window_gaps.py
   ```
5. **Steady state** (GPU; features committed): `run_e6_steady.py`, then
   `run_steady_suff.py`, then `run_steady_client_anchor.py`, as in the
   Section VI table; each step reads the folds the previous one built.

## 4. Hyper-parameters (identical for every trained condition)

```
--model fuse3_mamba_classifier --dataset_type byte_size_interval
--num_packet 5 --num_packet_byte 320 --seq_len 20 --stride_size 4
--batch_size 16 --epochs 30 --size_key signed_sizes --no_amp
--interval_encoding log10_decade
```

Each fold fine-tunes afresh from the public `fuse3_mamba.pth`. Under
`MASQUE_SELRULE=acc_loss` the epoch with the highest validation accuracy, ties
broken by the lowest validation loss, is evaluated once on that fold's test
split (`preds.json`); the first-maximum epoch of the original rule is dumped
alongside as `preds_acc_first.json`.

## 5. What is not in git and why

| Item | Size | Where |
|---|---|---|
| Raw captures (10 pcap files) | 3.7 GB | Zenodo (DOI pending); needed only for `split_*`, `ablate_constants.py`, `analyze_{header_geometry,peer_groups,flow_attrition}.py` |
| Day-0 paired capture (tunnel adapter + physical interface) | | not released; needed only for `analyze_protocol_encapsulation.py` and `diagnose_encapsulation_unmatched.py` |
| Per-flow pcaps `quic_vs_masque_bi/` | 15 MB | derived from the raw captures by step 1 of `README.md` |
| Condition and fold datasets under `data/processed/` other than the four committed | | deterministic transforms of the committed features (Section 3 above) |
| Fine-tuned checkpoints | 22.5 MB each | not kept; `verify_full_reproduced.py` checks that a re-run of the first release's `full` reproduces its committed predictions |
| Queue logs | | progress-bar output only; every fold's `log.txt` is kept |
| Pre-trained `fuse3_mamba.pth` | 117 MB | NetMamba authors' Google Drive (see README) |
