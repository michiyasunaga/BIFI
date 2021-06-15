exit 0;
################################################################################
# run the following commands one by one in the root directory of the repository
################################################################################
export PYTHONPATH=.

python src/c001__train_fixer.py --round_name round0 --gpu_id 0 --max_epoch 2
python src/c003__run_fixer.py   --round_name round0 --gpu_ids '0,1,2,3,4'
python src/c005__eval_fixer.py  --round_name round0
