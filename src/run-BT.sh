exit 0;
################################################################################
# run the following commands one by one in the root directory of the repository
################################################################################
export PYTHONPATH=.

#Round1
python src/c006__generate_paired_data_from_fixer.py --round_name round0 --out_round_name round1-BT-part1 --BIFI 0
python src/c002__train_breaker.py --round_name round1-BT-part1 --gpu_id 0 --max_epoch 3
python src/c004__run_breaker.py   --round_name round1-BT-part1 --gpu_ids '0,1,2,3,4'

python src/c007__generate_paired_data_from_breaker.py --round_name round1-BT-part1 --out_round_name round1-BT-part2 --BIFI 0
python src/c001__train_fixer.py --round_name round1-BT-part2 --gpu_id 0 --max_epoch 2 --continue_from 'data/round0/model-fixer/checkpoint.pt'
python src/c003__run_fixer.py   --round_name round1-BT-part2 --gpu_ids '0,1,2,3,4'
python src/c005__eval_fixer.py  --round_name round1-BT-part2


#Round2
python src/c006__generate_paired_data_from_fixer.py --round_name round1-BT-part2 --out_round_name round2-BT-part1 --BIFI 0
python src/c002__train_breaker.py --round_name round2-BT-part1 --gpu_id 0 --max_epoch 3 --continue_from 'data/round1-BT-part1/model-breaker/checkpoint.pt'
python src/c004__run_breaker.py   --round_name round2-BT-part1 --gpu_ids '0,1,2,3,4'

python src/c007__generate_paired_data_from_breaker.py --round_name round2-BT-part1 --out_round_name round2-BT-part2 --BIFI 0
python src/c001__train_fixer.py --round_name round2-BT-part2 --gpu_id 0 --max_epoch 2 --continue_from 'data/round1-BT-part2/model-fixer/checkpoint.pt'
python src/c003__run_fixer.py   --round_name round2-BT-part2 --gpu_ids '0,1,2,3,4'
python src/c005__eval_fixer.py  --round_name round2-BT-part2
