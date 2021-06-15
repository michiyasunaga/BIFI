import argparse
import hashlib
import numpy as np
import json, io, sys, os
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict, OrderedDict
from multiprocessing import Pool


from utils.code_error_checker import check_paren_error, check_ast_error
from utils.code_utils import preprocess_unk, code_toks_to_code_string, get_diff_metric
from utils.fairseq_utils import parse_fairseq_preds


def eval_one_pred_obj(pred_obj):
    #Deal with UNK
    _, unk_dict = preprocess_unk(pred_obj['code_toks_raw'])
    anonymize_dict = pred_obj['anonymize_dict']
    if anonymize_dict is None:
        anonymize_dict = {}
    anonymize_dict['<unk>'] = unk_dict
    anonymize_dict['<STRING>'] = []
    anonymize_dict['<COMMENT>'] = []
    #
    src = pred_obj['src'] #this is tok_format i.e. ' '.join(code_toks)
    src_code  = code_toks_to_code_string(src, anonymize_dict) #this is string_format
    ret_obj = {'progid': pred_obj['progid'],
                'anonymize_dict': pred_obj['anonymize_dict']
              }
    ret_obj['src']  = {'tok_format': src, 'string_format': src_code}
    #Get string_format from predicted code toks
    ret_obj['pred'] = []
    for pred in pred_obj['pred']:
        pred_code = code_toks_to_code_string(pred, anonymize_dict) #this is string_format
        res = check_paren_error(pred.split())
        if not isinstance(res, dict):
            res = check_ast_error(pred_code)
        diff_metric = get_diff_metric(src, pred)
        ret_obj['pred'].append({'tok_format': pred,
                                  'string_format': pred_code,
                                  'err_obj': res,
                                  'diff_metric': diff_metric})
    return ret_obj

def eval_one_split(pred_dir_prefix, split, pred_fname, goods, n_workers=80):
    pred_dir   = f'{pred_dir_prefix}{split}'
    pred_path = Path(f'{pred_dir}/{pred_fname}')
    preds = parse_fairseq_preds(str(pred_path))
    #load progids
    data_dir = 'data'
    progids = [l.strip() for l in open(f'{data_dir}/orig_good_code/orig.{split}.id')]
    assert len(preds) == len(progids)
    for j in range(len(preds)):
        progid = progids[j]
        preds[j]['progid'] = progid
        code_toks_raw = goods[progid].split()
        preds[j]['code_toks_raw'] = code_toks_raw
        preds[j]['anonymize_dict'] = None
    #
    print ('len(preds)', len(preds))
    with Pool(n_workers) as p:
        res = list(tqdm(p.imap(eval_one_pred_obj, preds), total=len(preds)))
    '''
      res: list of {'progid': ,  'anonymize_dict': ,
                    'src': {'tok_format': , 'string_format': },
                    'pred': {'tok_format':, 'string_format':, 'err_obj': }
                    }
    '''
    with open(f'{pred_path.parent}/{pred_path.stem}.evaluated.json', 'w') as f:
        json.dump(res, f, indent=2)


#BIFI version - uses critic to verify
def generate_paired_data_from_breaker_preds_for_BIFI(pred_dir_prefix, pred_fname, out_dir, load_dir):
    #Get new paired data
    train_data = {'good': [], 'bad': [], 'id': []}
    idxs_newdata_p, idxs_newdata_i, idxs_newdata_s = [], [], []
    cur_idx = 0
    for split in range(10):
      print ('split', split)
      pred_dir   = Path(f'{pred_dir_prefix}{split}')
      pred_path  = pred_dir/pred_fname
      pred_eval_path = f'{pred_path.parent}/{pred_path.stem}.evaluated.json'
      eval_objs = json.load(open(pred_eval_path))
      for eval_obj in tqdm(eval_objs):
        progid = eval_obj['progid']
        _data = {'good': [], 'bad': [], 'id': []}
        # print ('src\n', eval_obj['src']['string_format'])
        for k, pred_obj in enumerate(eval_obj['pred']):
          pred_err_obj = pred_obj['err_obj']
          diff_metric  = pred_obj['diff_metric']
          if (isinstance(pred_err_obj, dict)) and (0 < diff_metric <= 4): #keep broken examples
            name = '{:02d}-{}-{:03d}'.format(split, progid, k)
            src  = eval_obj['src']['tok_format'].strip()
            pred = pred_obj['tok_format'].strip()
            if 'unbalanced' in pred_err_obj['msg']:
              idxs_newdata_p.append(cur_idx)
            elif 'indent' in pred_err_obj['msg']:
              idxs_newdata_i.append(cur_idx)
            elif 'invalid syntax' == pred_err_obj['msg']:
              idxs_newdata_s.append(cur_idx)
            else:
              continue
            _data['id'  ].append(name)
            _data['bad'].append(pred) #pred is bad-side
            _data['good' ].append(src)
            cur_idx += 1
        train_data['id']   += _data['id']
        train_data['good'] += _data['good']
        train_data['bad']  += _data['bad']
    assert len(train_data['good']) == len(train_data['bad']) == len(train_data['id'])
    new_data_size = len(train_data['id'])
    assert new_data_size == len(idxs_newdata_p) + len(idxs_newdata_i) + len(idxs_newdata_s)
    print ('#new_data', new_data_size)
    print ('#new_data_p', len(idxs_newdata_p), '#new_data_i', len(idxs_newdata_i), '#new_data_s', len(idxs_newdata_s))
    # idxs_newdata = list(range(new_data_size))
    seed = (111 + int(hashlib.md5(str(out_dir).encode()).hexdigest(), 16)) % (2**31)
    print ('seed', seed)
    np.random.seed(seed)
    # np.random.shuffle(idxs_newdata); np.random.shuffle(idxs_newdata)
    np.random.shuffle(idxs_newdata_p); np.random.shuffle(idxs_newdata_p)
    np.random.shuffle(idxs_newdata_i); np.random.shuffle(idxs_newdata_i)
    np.random.shuffle(idxs_newdata_s); np.random.shuffle(idxs_newdata_s)
    #
    #Merge with the paired data in round0 and BIFI part1
    print ('loading round0 data')
    train_data_0 = {'good': [], 'bad': [], 'id': []}
    train_data_0['bad']  = [line.strip() for line in tqdm(open('data/round0/data_paired/train.bad'))]
    train_data_0['good'] = [line.strip() for line in tqdm(open('data/round0/data_paired/train.good'))]
    train_data_0['id']   = [line.strip() for line in tqdm(open('data/round0/data_paired/train.id'))]
    idxs_0 = list(range(len(train_data_0['id'])))
    np.random.shuffle(idxs_0); np.random.shuffle(idxs_0)
    print ('loading BIFI part1 data')
    train_data_1 = {'good': [], 'bad': [], 'id': []}
    train_data_1['bad']  = [line.strip() for line in tqdm(open(f'{load_dir}/train.bad'))]
    train_data_1['good'] = [line.strip() for line in tqdm(open(f'{load_dir}/train.good'))]
    train_data_1['id']   = [line.strip() for line in tqdm(open(f'{load_dir}/train.id'))]
    idxs_1 = list(range(len(train_data_1['id'])))
    np.random.shuffle(idxs_1); np.random.shuffle(idxs_1)
    assert len(set(train_data_1['id']).intersection(set(train_data_0['id']))) == 0
    assert len(set(train_data_1['id']).intersection(set(train_data['id']))) == 0
    assert len(set(train_data_0['id']).intersection(set(train_data['id']))) == 0
    #
    total_size = int(new_data_size * 2)
    _0_data_repeats  = (total_size//4)//len(idxs_0) +1
    _1_data_repeats  = (total_size*2//4)//len(idxs_1) +1
    new_data_p_repeats = (total_size*1//4//3)//len(idxs_newdata_p) +1
    new_data_i_repeats = (total_size*1//4//3)//len(idxs_newdata_i) +1
    new_data_s_repeats = (total_size*1//4//3)//len(idxs_newdata_s) +1
    idxs_0 = (idxs_0 * _0_data_repeats)[:total_size//4]
    idxs_1 = (idxs_1 * _1_data_repeats)[:total_size*2//4]
    idxs_newdata_p = (idxs_newdata_p * new_data_p_repeats)[:total_size*1//4//3]
    idxs_newdata_i = (idxs_newdata_i * new_data_i_repeats)[:total_size*1//4//3]
    idxs_newdata_s = (idxs_newdata_s * new_data_s_repeats)[:total_size*1//4//3]
    print ('combining all data..', total_size)
    idxs = [f'0_{i}' for i in idxs_0] + [f'1_{i}' for i in idxs_1] + [f'new_{i}' for i in idxs_newdata_p] + [f'new_{i}' for i in idxs_newdata_i] + [f'new_{i}' for i in idxs_newdata_s]
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
        elif _prefix == '1':
          fid.write(train_data_1['id'][_idx] +'\n')
          fgood.write(train_data_1['good'][_idx] +'\n')
          fbad.write(train_data_1['bad'][_idx] +'\n')
        else:
          fid.write(train_data['id'][_idx] +'\n')
          fgood.write(train_data['good'][_idx] +'\n')
          fbad.write(train_data['bad'][_idx] +'\n')
    os.system('cp {} {}'.format('data/round0/data_paired/dev.bad', out_dir))
    os.system('cp {} {}'.format('data/round0/data_paired/dev.good', out_dir))
    os.system('cp {} {}'.format('data/round0/data_paired/dev.id', out_dir))
    print ('done')

#backtranslation version - does not use critic
def generate_paired_data_from_breaker_preds_for_BT(pred_dir_prefix, pred_fname, out_dir, load_dir):
    #Get new paired data
    train_data = {'good': [], 'bad': [], 'id': []}
    idxs_newdata = []
    cur_idx = 0
    for split in range(10):
      print ('split', split)
      pred_dir   = Path(f'{pred_dir_prefix}{split}')
      pred_path  = pred_dir/pred_fname
      pred_eval_path = f'{pred_path.parent}/{pred_path.stem}.evaluated.json'
      eval_objs = json.load(open(pred_eval_path))
      for eval_obj in tqdm(eval_objs):
        progid = eval_obj['progid']
        _data = {'good': [], 'bad': [], 'id': []}
        # print ('src\n', eval_obj['src']['string_format'])
        for k, pred_obj in enumerate(eval_obj['pred']):
          pred_err_obj = pred_obj['err_obj']
          diff_metric  = pred_obj['diff_metric']
          pred = pred_obj['tok_format'].strip()
          # if (isinstance(pred_err_obj, dict)) and (0 < diff_metric <= 4): #keep broken examples
          if len(pred.split()) <= 120:
            name = '{:02d}-{}-{:03d}'.format(split, progid, k)
            src  = eval_obj['src']['tok_format'].strip()
            idxs_newdata.append(cur_idx)
            _data['id'  ].append(name)
            _data['bad' ].append(pred) #pred is bad-side
            _data['good'].append(src)
            cur_idx += 1
        train_data['id']   += _data['id']
        train_data['good'] += _data['good']
        train_data['bad']  += _data['bad']
    assert len(train_data['good']) == len(train_data['bad']) == len(train_data['id'])
    new_data_size = len(train_data['id'])
    assert new_data_size == len(idxs_newdata)
    print ('#new_data', new_data_size)
    # idxs_newdata = list(range(new_data_size))
    seed = (111 + int(hashlib.md5(str(out_dir).encode()).hexdigest(), 16)) % (2**31)
    print ('seed', seed)
    np.random.seed(seed)
    np.random.shuffle(idxs_newdata); np.random.shuffle(idxs_newdata)
    #
    #Merge with the paired data in round0 and BIFI part1
    print ('loading round0 data')
    train_data_0 = {'good': [], 'bad': [], 'id': []}
    train_data_0['bad']  = [line.strip() for line in tqdm(open('data/round0/data_paired/train.bad'))]
    train_data_0['good'] = [line.strip() for line in tqdm(open('data/round0/data_paired/train.good'))]
    train_data_0['id']   = [line.strip() for line in tqdm(open('data/round0/data_paired/train.id'))]
    idxs_0 = list(range(len(train_data_0['id'])))
    np.random.shuffle(idxs_0); np.random.shuffle(idxs_0)
    # print ('loading BIFI part1 data')
    train_data_1 = {'good': [], 'bad': [], 'id': []}
    # train_data_1['bad']  = [line.strip() for line in tqdm(open(f'{load_dir}/train.bad'))]
    # train_data_1['good'] = [line.strip() for line in tqdm(open(f'{load_dir}/train.good'))]
    # train_data_1['id']   = [line.strip() for line in tqdm(open(f'{load_dir}/train.id'))]
    idxs_1 = list(range(len(train_data_1['id'])))
    np.random.shuffle(idxs_1); np.random.shuffle(idxs_1)
    assert len(set(train_data_1['id']).intersection(set(train_data_0['id']))) == 0
    assert len(set(train_data_1['id']).intersection(set(train_data['id']))) == 0
    assert len(set(train_data_0['id']).intersection(set(train_data['id']))) == 0
    #
    total_size = 30_000_000
    _0_data_repeats  = (total_size//2)//len(idxs_0) +1
    _1_data_repeats  = 0
    new_data_repeats = (total_size//2)//len(idxs_newdata) +1
    idxs_0 = (idxs_0 * _0_data_repeats)[:total_size//2]
    idxs_1 = []
    idxs_newdata = (idxs_newdata * new_data_repeats)[:total_size//2]
    print ('combining all data..', total_size)
    idxs = [f'0_{i}' for i in idxs_0] + [f'1_{i}' for i in idxs_1] + [f'new_{i}' for i in idxs_newdata]
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
        elif _prefix == '1':
          fid.write(train_data_1['id'][_idx] +'\n')
          fgood.write(train_data_1['good'][_idx] +'\n')
          fbad.write(train_data_1['bad'][_idx] +'\n')
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
pred_dir_root = Path(args.pred_dir_root) if args.pred_dir_root else round_dir/'orig_good'
pred_dir_prefix = str(pred_dir_root/'fairseq_preprocess__orig_good.')
pred_fname  = 'model-breaker.pred.txt'
goods = json.load(open(data_dir/'orig_good_code/orig.good.json'))


n_splits = 10  #all the original good code is split into 10 chunks for faster processing
for split in range(n_splits):
    eval_one_split(pred_dir_prefix, split, pred_fname, goods, n_workers=72)


load_dir = data_dir/args.round_name/'data_paired_pure'
out_dir  = data_dir/args.out_round_name/'data_paired'
if args.BIFI:
    generate_paired_data_from_breaker_preds_for_BIFI(pred_dir_prefix, pred_fname, out_dir, load_dir)
else:
    generate_paired_data_from_breaker_preds_for_BT(pred_dir_prefix, pred_fname, out_dir, load_dir)
