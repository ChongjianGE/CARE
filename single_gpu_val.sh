export PYTHONPATH=$PYTHONPATH:/apdcephfs/share_1290939/chongjiange/github_repo/CARE/
now=$(date +"%Y%m%d_%H%M%S")
echo EXP-$now

python3.6 -m torch.distributed.launch --nproc_per_node=1 \
  tools/eval_new.py \
  --distributed \
  -b 256 \
  -ckpt 100 \
  --experiment_name debug \
  -f exps/arxiv/linear_eval_exp_care.py
