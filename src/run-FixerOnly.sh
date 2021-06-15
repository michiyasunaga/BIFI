exit 0;
################################################################################
# run the following commands one by one in the root directory of the repository
################################################################################
export PYTHONPATH=.

#Round1
python src/c006__generate_paired_data_from_fixer.py --round_name round0 --out_round_name round1-FixerOnly
python src/c001__train_fixer.py --round_name round1-FixerOnly --gpu_id 0 --max_epoch 1 --continue_from 'data/round0/model-fixer/checkpoint.pt'
python src/c003__run_fixer.py   --round_name round1-FixerOnly --gpu_ids '0,1,2,3,4'
python src/c005__eval_fixer.py  --round_name round1-FixerOnly


#Round2
python src/c006__generate_paired_data_from_fixer.py --round_name round1-FixerOnly --out_round_name round2-FixerOnly
python src/c001__train_fixer.py --round_name round2-FixerOnly --gpu_id 0 --max_epoch 1 --continue_from 'data/round1-FixerOnly/model-fixer/checkpoint.pt'
python src/c003__run_fixer.py   --round_name round2-FixerOnly --gpu_ids '0,1,2,3,4'
python src/c005__eval_fixer.py  --round_name round2-FixerOnly
