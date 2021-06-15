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
from fairseq_utils import parse_fairseq_preds


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
                'orig_err_obj': pred_obj['orig_err_obj'],
                'anonymize_dict': pred_obj['anonymize_dict']
              }
    ret_obj['src']  = {'tok_format': src, 'string_format': src_code}
    #Get string_format from predicted code toks
    ret_obj['pred'] = []
    for pred in pred_obj['pred']:
        pred_code = code_toks_to_code_string(pred, anonymize_dict) #this is string_format
        orig_err_obj = pred_obj['orig_err_obj']
        if orig_err_obj['msg'] == 'unbalanced (){}[]':
            #NOTE: `pred` is tok_format i.e. ' '.join(code_toks)
            res = check_paren_error(pred.split())
        else:
            res = check_ast_error(pred_code)
        diff_metric = get_diff_metric(src, pred)
        ret_obj['pred'].append({'tok_format': pred,
                                  'string_format': pred_code,
                                  'err_obj': res,
                                  'diff_metric': diff_metric})
    return ret_obj

def eval_one_split(pred_dir_prefix, split, pred_fname, n_workers=80):
    pred_dir   = f'{pred_dir_prefix}{split}'
    pred_path = Path(f'{pred_dir}/{pred_fname}')
    preds = parse_fairseq_preds(str(pred_path))
    #load progids
    data_dir = 'data'
    progids = [l.strip() for l in open(f'{data_dir}/orig_bad_code/orig.{split}.id')]
    assert len(preds) == len(progids)
    #load original err_obj
    bads = json.load(open(f'{data_dir}/orig_bad_code/orig.bad.json'))
    for j in range(len(preds)):
        progid = progids[j]
        preds[j]['progid'] = progid
        preds[j]['orig_err_obj'] = bads[progid]['err_obj']
        code_toks_raw = bads[progid]['code_toks_joined'].split()
        anonymize_dict = bads[progid]['anonymize_dict']
        if 'window_span' in bads[progid]:
            ws = bads[progid]['window_span']
            code_toks_raw = code_toks_raw[ws[0]:ws[1]]
            anonymize_dict = None
        preds[j]['code_toks_raw'] = code_toks_raw
        preds[j]['anonymize_dict'] = anonymize_dict
    #
    print ('len(preds)', len(preds))
    with Pool(n_workers) as p:
        res = list(tqdm(p.imap(eval_one_pred_obj, preds), total=len(preds)))
    '''
      res: list of {'progid': , 'orig_err_obj': , 'anonymize_dict': ,
                    'src': {'tok_format': , 'string_format': },
                    'pred': {'tok_format':, 'string_format':, 'err_obj': }
                    }
    '''
    with open(f'{pred_path.parent}/{pred_path.stem}.evaluated.json', 'w') as f:
        json.dump(res, f, indent=2)

def get_test_result(pred_dir_prefix, pred_fname):
    #
    def collate_eval():
      success  = []; denom = 0
      success_by_group = defaultdict(list); denom_by_group = defaultdict(int)
      agg_obj = {}
      for split in {3,4}: #heldout test set
        print ('split', split)
        pred_dir   = Path(f'{pred_dir_prefix}{split}')
        pred_path  = pred_dir/pred_fname
        pred_eval_path = f'{pred_path.parent}/{pred_path.stem}.evaluated.json'
        eval_objs = json.load(open(pred_eval_path))
        for eval_obj in eval_objs:
          progid = eval_obj['progid']
          orig_err_type = eval_obj['orig_err_obj']['msg']
          if 'indent' in orig_err_type:
              orig_err_type = 'indentation error'
          denom += 1
          denom_by_group[orig_err_type] += 1
          for k, pred_obj in enumerate(eval_obj['pred']):
            pred_err_obj = pred_obj['err_obj']
            diff_metric  = pred_obj['diff_metric']
            if (pred_err_obj == 0) and (0 < diff_metric <= 4):
              name = '{:02d}-{}-{:03d}'.format(split, progid, k)
              success.append(name)
              success_by_group[orig_err_type].append(name)
      return success, denom, success_by_group, denom_by_group
    #
    def print_stats(name_list, _denom):
      top1 = set()
      for name in name_list:
        split, progid, k = name.split('-')
        if int(split) in {3,4}: #test set
          if int(k)==0:
            top1.add(f'{split}-{progid}')
      acc = len(top1)/float(_denom)*100
      print ('   acc: {} ({:.1f}%) | denom {}'.format(len(top1), acc, _denom))
      return acc
    #
    success, denom, success_by_group, denom_by_group = collate_eval()
    acc_dict = {}
    print ('Total'); acc = print_stats(success, denom); acc_dict['total'] = acc
    print ('-'*50)
    for err_type in success_by_group:
        print (f'{err_type.capitalize()}')
        acc = print_stats(success_by_group[err_type], denom_by_group[err_type])
        acc_dict[err_type] = acc
    json.dump(acc_dict, open(Path(pred_dir_prefix).parent/'stats.json', 'w'), indent=2)



#################################### main #####################################
parser = argparse.ArgumentParser()
parser.add_argument('--round_name')
parser.add_argument('--pred_dir_root', default='')
args = parser.parse_args()

data_dir = Path('data')
round_dir = data_dir/args.round_name
pred_dir_root = Path(args.pred_dir_root) if args.pred_dir_root else round_dir/'orig_bad'
pred_dir_prefix = str(pred_dir_root/'fairseq_preprocess__orig_bad.')
pred_fname  = 'model-fixer.pred.txt'


n_splits = 5  #all the original bad code is split into 5 chunks for faster processing
for split in range(n_splits):
    eval_one_split(pred_dir_prefix, split, pred_fname, n_workers=10)

get_test_result(pred_dir_prefix, pred_fname)
