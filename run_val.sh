export PYTHONPATH=$PYTHONPATH:/apdcephfs/share_1290939/chongjiange/github_repo/CARE/
now=$(date +"%Y%m%d_%H%M%S")
echo EXP-$now

python3.6 -m torch.distributed.launch --nproc_per_node=8 \
  tools/eval_new.py \
  --distributed \
  -b 256 \
  -ckpt 100 \
  --experiment-name care_res50_100e_github \
  -f exps/arxiv/linear_eval_exp_care.py
