now=$(date +"%Y%m%d_%H%M%S")
echo EXP-$now

python3.6 -m torch.distributed.launch --nproc_per_node=1 \
  care/tools/eval_new.py \
  --distributed \
  -b 256 \
  -ckpt 100 \
  --experiment_name byol_parallel_5bot_syn_lamda100_detach_2fc_20210516_143302 \
  -f care/exps/arxiv/linear_eval_exp_care.py
