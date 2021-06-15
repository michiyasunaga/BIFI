import numpy as np
import json
from collections import OrderedDict
from pathlib import Path

ROOT = Path('data')
ROOT.mkdir(exist_ok=True)


bads = json.load(open(ROOT/'orig_bad_code/orig.bad.json'), object_pairs_hook=OrderedDict)
bads_keys = list(bads.keys())
_len = len(bads)
csize = _len//5 +1
for c in range(5):
    with open(str(ROOT/f'orig_bad_code/orig.{c}.id'), 'w') as f1, open(str(ROOT/f'orig_bad_code/orig.{c}.bad'), 'w') as f2:
        for _id in bads_keys[csize*c: csize*(c+1)]:
            print (_id, file=f1)
            code_toks_raw = bads[_id]['code_toks_joined'].split()
            if 'window_span' in bads[_id]:
                start, end = bads[_id]['window_span']
                code_toks_raw = code_toks_raw[start:end]
            code = ' '.join(code_toks_raw)
            print (code, file=f2)


goods = json.load(open(ROOT/'orig_good_code/orig.good.json'), object_pairs_hook=OrderedDict)
goods_keys = list(goods.keys())
_len = len(goods)
csize = _len//10 +1
for c in range(10):
    with open(str(ROOT/f'orig_good_code/orig.{c}.id'), 'w') as f1, open(str(ROOT/f'orig_good_code/orig.{c}.good'), 'w') as f2:
        for _id in goods_keys[csize*c: csize*(c+1)]:
            print (_id, file=f1)
            print (goods[_id]['code_toks_joined'], file=f2)
