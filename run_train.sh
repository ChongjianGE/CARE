export PYTHONPATH=$PYTHONPATH:/apdcephfs/share_1290939/chongjiange/github_repo/CARE/
now=$(date +"%Y%m%d_%H%M%S")
echo EXP-$now

python3.6 -m torch.distributed.launch --nproc_per_node=8 \
	tools/train_new.py \
	--distributed \
	-b 1024 \
	-d 0-7 \
	-n 2 \
	--experiment-name care_res50_100e_github \
	-f exps/arxiv/exp_8_v100/care_100e_exp.py

