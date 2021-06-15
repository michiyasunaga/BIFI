import io
import sys
import json, os, re
import token
import numpy as np
import editdistance
from collections import defaultdict, OrderedDict, Counter
from copy import deepcopy

sys.path.insert(0, 'utils')
from code_tokenizer import tokenize


vocab_file = 'data/token_vocab.txt'
vocab = set([line.split()[0] for line in open(vocab_file)])

def toks2lines(code_toks_raw):
    lines = []
    line = []
    for tok in code_toks_raw:
        if tok in {'<NL>', '<NEWLINE>'}:
            line.append(tok)
            lines.append(line); line = []
        else:
            line.append(tok)
    if line:
        lines.append(line)
    return lines

def preprocess_unk(code_toks_raw):
    ret = []
    unk_dict = []
    for tok in code_toks_raw:
        if tok not in vocab:
            ret.append('<unk>')
            unk_dict.append(tok)
        else:
            ret.append(tok)
    return ret, unk_dict

def code_toks_to_code_string(code_toks_joined, anonymize_dict=None):
    if anonymize_dict is not None:
        anonymize_dict = deepcopy(anonymize_dict)

    default_replace_map = {'<unk>': 'unk',
                          '<COMMENT>': '##',
                          '<NUMBER>' : '1',
                          '<STRING>' : '"str"'}
    cur_indent  = 0
    indent_unit = '    '
    toks = code_toks_joined.split()
    final_toks = []
    startline = True
    for tok in toks:
        tok_post = None
        if tok == '<INDENT>':
            cur_indent += 1
            continue
        elif tok == '<DEDENT>':
            cur_indent -= 1
            continue
        elif tok in ('<NEWLINE>', '<NL>'):
            final_toks.append('\n')
            startline = True
            continue
        if startline:
            cur_indent = max(0, cur_indent)
            final_toks.append(indent_unit * cur_indent)
            startline = False
        else:
            final_toks.append(' ')
        if (anonymize_dict is not None) and (tok in anonymize_dict) and (len(anonymize_dict[tok]) > 0):
            tok_post = anonymize_dict[tok].pop(0)
        else:
            tok_post = default_replace_map[tok] if tok in default_replace_map else tok
        final_toks.append(tok_post)
    code = "".join(final_toks)
    return code

def get_diff_metric(src, pred):
    src_toks  = src.split()
    pred_toks = pred.split()
    diff_metric = editdistance.eval(src_toks, pred_toks)
    return diff_metric



def tokenize_python_code(code_string):
    try:
        tokens = tokenize(io.BytesIO(code_string.encode('utf8')).readline)
        toks = [t for t in tokens]
    except Exception as e:
        print (e)
        return 1
    SPECIAL = {'STRING', 'COMMENT', 'INDENT', 'DEDENT', 'NEWLINE', 'NL'}
    IGNORE = {'ENCODING', 'ENDMARKER'}
    toks_raw = []
    anonymize_dict = defaultdict(list)
    for tok in toks:
        tok_type_name = token.tok_name[tok.type]
        if tok_type_name in IGNORE:
            continue
        elif tok_type_name in SPECIAL:
            toks_raw.append(f'<{tok_type_name}>')
            if tok_type_name in {'STRING', 'COMMENT'}:
                anonymize_dict[f'<{tok_type_name}>'].append(tok.string)
        else:
            toks_raw.append(tok.string)
    assert len(toks_raw) == len(toks)-2
    return toks_raw, anonymize_dict
