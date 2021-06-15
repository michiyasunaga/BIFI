import argparse
import hashlib
import numpy as np
import json, io, sys, os
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict, OrderedDict
from multiprocessing import Pool


#BIFI version - uses critic to verify
def generate_paired_data_from_fixer_preds_for_BIFI(pred_dir_prefix, pred_fname, out_dir):
    #Get new paired data
    train_data = {'good': [], 'bad': [], 'id': []}
    for split in {0,1,2}: #available for training
      print ('split', split)
      pred_dir   = Path(f'{pred_dir_prefix}{split}')
      pred_path  = pred_dir/pred_fname
      pred_eval_path = f'{pred_path.parent}/{pred_path.stem}.evaluated.json'
      eval_objs = json.load(open(pred_eval_path))
      for eval_obj in eval_objs:
        progid = eval_obj['progid']
        for k, pred_obj in enumerate(eval_obj['pred']):
          pred_err_obj = pred_obj['err_obj']
          diff_metric  = pred_obj['diff_metric']
          if (pred_err_obj == 0) and (0 < diff_metric <= 4):
            name = '{:02d}-{}-{:03d}'.format(split, progid, k)
            src  = eval_obj['src']['tok_format'].strip()
            pred = pred_obj['tok_format'].strip()
            train_data['id'  ].append(name)
            train_data['good'].append(pred)
            train_data['bad' ].append(src)
    assert len(train_data['good']) == len(train_data['bad']) == len(train_data['id'])
    new_data_size = len(train_data['id'])
    print ('#new_data', new_data_size)
    os.system(f'mkdir -p {out_dir}_pure')
    with open(f'{out_dir}_pure/train.id', 'w') as fid, \
         open(f'{out_dir}_pure/train.good', 'w') as fgood, \
         open(f'{out_dir}_pure/train.bad', 'w') as fbad:
      for _idx in tqdm(range(new_data_size)):
        fid.write(train_data['id'][_idx] +'\n')
        fgood.write(train_data['good'][_idx] +'\n')
        fbad.write(train_data['bad'][_idx] +'\n')
    idxs_newdata = list(range(new_data_size))
    #
    #Merge with round0 paired data
    print ('loading round0 data')
    train_data_0 = {'good': [], 'bad': [], 'id': []}
    train_data_0['bad']  = [line.strip() for line in tqdm(open('data/round0/data_paired/train.bad'))]
    train_data_0['good'] = [line.strip() for line in tqdm(open('data/round0/data_paired/train.good'))]
    train_data_0['id']   = [line.strip() for line in tqdm(open('data/round0/data_paired/train.id'))]
    idxs_0 = list(range(len(train_data_0['id'])))
    seed = (111 + int(hashlib.md5(str(out_dir).encode()).hexdigest(), 16)) % (2**31)
    print ('seed', seed)
    np.random.seed(seed)
    np.random.shuffle(idxs_0); np.random.shuffle(idxs_0)
    total_size = 30_000_000
    _0_data_repeats  = (total_size//3)//len(idxs_0) +1
    new_data_repeats = (total_size*2//3)//new_data_size +1
    idxs_0 = (idxs_0 * _0_data_repeats)[:total_size//3]
    idxs_newdata = idxs_newdata * new_data_repeats
    print ('combining all data')
    idxs = [f'0_{i}' for i in idxs_0] + [f'new_{i}' for i in idxs_newdata]
    np.random.shuffle(idxs); np.random.shuffle(idxs)
    #
    #Write out data
    os.system(f'mkdir -p {out_dir}')
    print ('writing out data')
    with open(f'{out_dir}/train.id', 'w') as fid, \
         open(f'{out_dir}/train.good', 'w') as fgood, \
         open(f'{out_dir}/train.bad', 'w') as fbad:
      for idx in tqdm(idxs):
        _prefix, _idx = idx.split('_')
        _idx = int(_idx)
        if _prefix == '0':
          fid.write(train_data_0['id'][_idx] +'\n')
          fgood.write(train_data_0['good'][_idx] +'\n')
          fbad.write(train_data_0['bad'][_idx] +'\n')
        else:
          fid.write(train_data['id'][_idx] +'\n')
          fgood.write(train_data['good'][_idx] +'\n')
          fbad.write(train_data['bad'][_idx] +'\n')
    os.system('cp {} {}'.format('data/round0/data_paired/dev.bad', out_dir))
    os.system('cp {} {}'.format('data/round0/data_paired/dev.good', out_dir))
    os.system('cp {} {}'.format('data/round0/data_paired/dev.id', out_dir))
    print ('done')


#backtranslation version - does not use critic
def generate_paired_data_from_fixer_preds_for_BT(pred_dir_prefix, pred_fname, out_dir):
    #Get new paired data
    train_data = {'good': [], 'bad': [], 'id': []}
    for split in {0,1,2}: #available for training
      print ('split', split)
      pred_dir   = Path(f'{pred_dir_prefix}{split}')
      pred_path  = pred_dir/pred_fname
      pred_eval_path = f'{pred_path.parent}/{pred_path.stem}.evaluated.json'
      eval_objs = json.load(open(pred_eval_path))
      for eval_obj in eval_objs:
        progid = eval_obj['progid']
        for k, pred_obj in enumerate(eval_obj['pred']):
          # pred_err_obj = pred_obj['err_obj']
          # diff_metric  = pred_obj['diff_metric']
          pred = pred_obj['tok_format'].strip()
          # if (pred_err_obj == 0) and (0 < diff_metric <= 4):
          if len(pred.split()) <= 120:
            name = '{:02d}-{}-{:03d}'.format(split, progid, k)
            src  = eval_obj['src']['tok_format'].strip()
            train_data['id'  ].append(name)
            train_data['good'].append(pred)
            train_data['bad' ].append(src)
    assert len(train_data['good']) == len(train_data['bad']) == len(train_data['id'])
    new_data_size = len(train_data['id'])
    print ('#new_data', new_data_size)
    os.system(f'mkdir -p {out_dir}_pure')
    with open(f'{out_dir}_pure/train.id', 'w') as fid, \
         open(f'{out_dir}_pure/train.good', 'w') as fgood, \
         open(f'{out_dir}_pure/train.bad', 'w') as fbad:
      for _idx in tqdm(range(new_data_size)):
        fid.write(train_data['id'][_idx] +'\n')
        fgood.write(train_data['good'][_idx] +'\n')
        fbad.write(train_data['bad'][_idx] +'\n')
    idxs_newdata = list(range(new_data_size))
    #
    #Merge with round0 paired data
    print ('loading round0 data')
    train_data_0 = {'good': [], 'bad': [], 'id': []}
    train_data_0['bad']  = [line.strip() for line in tqdm(open('data/round0/data_paired/train.bad'))]
    train_data_0['good'] = [line.strip() for line in tqdm(open('data/round0/data_paired/train.good'))]
    train_data_0['id']   = [line.strip() for line in tqdm(open('data/round0/data_paired/train.id'))]
    idxs_0 = list(range(len(train_data_0['id'])))
    seed = (111 + int(hashlib.md5(str(out_dir).encode()).hexdigest(), 16)) % (2**31)
    np.random.seed(seed)
    np.random.shuffle(idxs_0); np.random.shuffle(idxs_0)
    total_size = 30_000_000
    _0_data_repeats  = (total_size//3)//len(idxs_0) +1
    new_data_repeats = (total_size*2//3)//new_data_size +1
    idxs_0 = (idxs_0 * _0_data_repeats)[:total_size//3]
    idxs_newdata = idxs_newdata * new_data_repeats
    print ('combining all data')
    idxs = [f'0_{i}' for i in idxs_0] + [f'new_{i}' for i in idxs_newdata]
    np.random.shuffle(idxs); np.random.shuffle(idxs)
    #
    #Write out data
    os.system(f'mkdir -p {out_dir}')
    print ('writing out data')
    with open(f'{out_dir}/train.id', 'w') as fid, \
         open(f'{out_dir}/train.good', 'w') as fgood, \
         open(f'{out_dir}/train.bad', 'w') as fbad:
      for idx in tqdm(idxs):
        _prefix, _idx = idx.split('_')
        _idx = int(_idx)
        if _prefix == '0':
          fid.write(train_data_0['id'][_idx] +'\n')
          fgood.write(train_data_0['good'][_idx] +'\n')
          fbad.write(train_data_0['bad'][_idx] +'\n')
        else:
          fid.write(train_data['id'][_idx] +'\n')
          fgood.write(train_data['good'][_idx] +'\n')
          fbad.write(train_data['bad'][_idx] +'\n')
    os.system('cp {} {}'.format('data/round0/data_paired/dev.bad', out_dir))
    os.system('cp {} {}'.format('data/round0/data_paired/dev.good', out_dir))
    os.system('cp {} {}'.format('data/round0/data_paired/dev.id', out_dir))
    print ('done')



#################################### main #####################################
parser = argparse.ArgumentParser()
parser.add_argument('--round_name')
parser.add_argument('--out_round_name')
parser.add_argument('--pred_dir_root', default='')
parser.add_argument('--BIFI', type=int, default=1)
args = parser.parse_args()


data_dir = Path('data')
round_dir = data_dir/args.round_name
pred_dir_root = Path(args.pred_dir_root) if args.pred_dir_root else round_dir/'orig_bad'
pred_dir_prefix = str(pred_dir_root/'fairseq_preprocess__orig_bad.')
pred_fname  = 'model-fixer.pred.txt'


out_dir = data_dir/args.out_round_name/'data_paired'
if args.BIFI:
    generate_paired_data_from_fixer_preds_for_BIFI(pred_dir_prefix, pred_fname, out_dir)
else:
    generate_paired_data_from_fixer_preds_for_BT(pred_dir_prefix, pred_fname, out_dir)
