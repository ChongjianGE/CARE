export PYTHONPATH=$PYTHONPATH:/apdcephfs/share_1290939/chongjiange/github_repo/CARE/
now=$(date +"%Y%m%d_%H%M%S")
echo EXP-$now

python -m torch.distributed.launch --nproc_per_node=1 --master_port 29501 \
	tools/train_new.py \
	--distributed \
	-b 128 \
	-d 0-7 \
	-n 2 \
	--experiment-name debug \
	-f exps/arxiv/exp_8_v100/care_100e_exp.py
