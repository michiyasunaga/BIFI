exit 0;
################################################################################
# run the following commands one by one in the root directory of the repository
################################################################################
export PYTHONPATH=.

python src/c001__train_fixer.py --round_name round0 --gpu_id 0 --max_epoch 2
python src/c003__run_fixer.py   --round_name round0 --gpu_ids '0,1,2,3,4'
python src/c005__eval_fixer.py  --round_name round0


According to the paper, the initialization step contains the training of an initial fixer and a breaker as well. See the image in PR's description.
However, in this script created for initialization step, you do not train the initial breaker. 
The first breaker is trained in the first round.

Is this a bug?
