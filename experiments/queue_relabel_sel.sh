#!/usr/bin/env bash
# [2026-10-12 ICOIN r2 §7.5] 改标版 (11 条流 -> Direct) 全部 NetMamba+ 实验按新选模规则 acc_loss 重跑。
# 见 docs/01_Planning/Handoff_Relabel_SelRule_Rerun_2026-10-12.md §6。
# 范围 = queue_icoin_r2_relabel.sh 的 NetMamba+ 部分; RF / 经典模型 / 零训练规则没有 epoch 选择, 不重跑。
# 开关 MASQUE_SELRULE=acc_loss (experiments/selrule_util.py): 训练入口换 finetune_selrule.py,
# 主结果用 ckpt-acc_loss.pth, 输出目录一律加 _sel 后缀, 已有结果只读。
# 前置: 冒烟测试 (只跑 full) 已通过 —— full 在本队列里会被自动跳过。
# 用法 (WSL MAMBA):  bash experiments/queue_relabel_sel.sh > outputs/relabel_sel/queue.log 2>&1
# 每步打印 "STEP <name>" / "DONE <name>" / "FAIL <name> rc=<rc>"; 某步失败不阻塞后续步骤。
# 中断后续跑: START_AT=<步骤名> bash experiments/queue_relabel_sel.sh >> ... 2>&1
#   之前的步骤打印 SKIP 不执行; 该步内已完成的折由各 runner 自动跳过, 半截的折清空重训。

cd "$(dirname "$0")/.." || exit 1
export MASQUE_SELRULE=acc_loss
started=${START_AT:+0}   # 设了 START_AT 时为 0 (尚未到达), 否则为空 (全部执行)

run() {   # run <name> <command...>
  local name=$1; shift
  if [ "$started" = 0 ] && [ "$name" != "$START_AT" ]; then echo "SKIP $name"; return; fi
  started=1
  echo "STEP $name  $(date '+%H:%M:%S')  gpu=$(nvidia-smi --query-gpu=clocks.sm,pstate --format=csv,noheader 2>/dev/null)"
  "$@"
  local rc=$?
  if [ $rc -eq 0 ]; then echo "DONE $name  $(date '+%H:%M:%S')"; else echo "FAIL $name rc=$rc"; fi
}

# 开关会把 --out_root outputs/e3r_kfold 改写成 outputs/e3r_kfold_sel
E3R="--fold_root data/processed/e3r_folds --out_root outputs/e3r_kfold"

# 1. 全特征 (冒烟测试已跑; 保留 ckpt-acc_loss.pth: 表 II 的 clean 检测器要用)
run sel_full          python -u experiments/run_kfold_e3.py $E3R --only full --keep-ckpt
# 2. 表 I 的 8 行
run sel_table1        python -u experiments/run_kfold_e3.py $E3R --only no_intervals,no_direction,no_bytes,no_sizes,suff_mag,suff_dir,nosizes_nobytes,suff_bytes
# 3. 表 II (复用 e6r_adv_folds / e6r_adv_evals, 不重造数据)
run sel_e6r_adv       python -u experiments/run_e6_adv_kfold.py --relabel --stage all
# 4. #1 / #2 / C 系列 (9 个条件)
run sel_icoin_r2      python -u experiments/run_icoin_r2.py --labels relabel
# 5. 常量消融 (VI-C)
run sel_const         python -u experiments/run_kfold_e3.py $E3R --only c1,c2,c3,c123,c123_nobytes
# 6. 抖动 (IV-C)
run sel_jitter        python -u experiments/run_kfold_e3.py $E3R --only jit50us,jit200us,jit1ms,jit5ms,jit1ms_full
# 7. E4 Cloudflare 留出 / leave-one-/16-out (VI-B)
run sel_e4            env MASQUE_RELABEL=1 python -u experiments/run_e4_cloudflare.py
# 8. CPU: 工作点 (ROC-AUC、低 FPR)
run sel_opoints       env MASQUE_RELABEL=1 python -u experiments/analyze_e6_operating_points.py --condition full

echo "QUEUE_END $(date '+%H:%M:%S')"
