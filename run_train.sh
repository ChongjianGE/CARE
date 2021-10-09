now=$(date +"%Y%m%d_%H%M%S")

echo EXP-$now

python3.6 -m torch.distributed.launch --nproc_per_node=8 \
	care/tools/train_new.py \
	--distributed \
	-b 1024 \
	-d 0-7 \
	-n 2 \
	--experiment-name care_5trans_lambda100_$now \
	-f care/exps/arxiv/exp_8_v100/care_100e_exp.py

