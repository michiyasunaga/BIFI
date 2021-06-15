import shlex
import shutil
import subprocess
import json, os
from pathlib import Path
from collections import defaultdict, OrderedDict

def fairseq_preprocess(src, tgt, destdir, trainpref=None, validpref=None, testpref=None, srcdict=None, **kwargs):
    additional_cmds = ''.join([f"--{k.replace('_', '-')} {v} " for k, v in kwargs.items() if not isinstance(v, bool)])
    additional_cmds += ''.join([f"--{k.replace('_', '-')} " for k, v in kwargs.items() if isinstance(v, bool) and v])
    cmd = f'fairseq-preprocess --source-lang {src} --destdir {destdir} \
            --joined-dictionary --workers 50 '
    if tgt is not None:
        cmd += f'--target-lang {tgt} '
    if trainpref is not None:
        cmd += f'--trainpref {trainpref} '
    if validpref is not None:
        cmd += f'--validpref {validpref} '
    if testpref is not None:
        cmd += f'--testpref {testpref} '
    if srcdict is not None:
        cmd += f'--srcdict {srcdict} '
    cmd += additional_cmds
    subprocess.run(shlex.split(cmd))

def fairseq_train(GPUs, preprocess_dir, save_dir, logfile, src, tgt, model='transformer',
                  criterion='label_smoothed_cross_entropy',
                  encoder_layers=4, decoder_layers=4, encoder_embed_dim=256,
                  decoder_embed_dim=256, encoder_ffn_embed_dim=1024,
                  decoder_ffn_embed_dim=1024, encoder_attention_heads=8,
                  decoder_attention_heads=8, dropout=0.4,
                  attention_dropout=0.2, relu_dropout=0.2,
                  weight_decay=0.0001, warmup_updates=400, warmup_init_lr=1e-4,
                  lr=1e-3, min_lr=1e-9, max_tokens=1000, update_freq=4,
                  max_epoch=10, save_interval=1, log_interval=100, log_format='tqdm',
                  user_dir=None, reset=False, restore_file=None, **kwargs):
    if True:
        additional_cmds = ''.join([f"--{k.replace('_', '-')} {v} " for k, v in kwargs.items() if not isinstance(v, bool)])
        additional_cmds += ''.join([f"--{k.replace('_', '-')} " for k, v in kwargs.items() if isinstance(v, bool) and v])
        cmd = f"fairseq-train \
                {preprocess_dir} \
               --source-lang {src} --target-lang {tgt} \
               --arch {model} --share-all-embeddings \
               --encoder-layers {encoder_layers} --decoder-layers {decoder_layers} \
               --encoder-embed-dim {encoder_embed_dim} --decoder-embed-dim {decoder_embed_dim} \
               --encoder-ffn-embed-dim {encoder_ffn_embed_dim} --decoder-ffn-embed-dim {decoder_ffn_embed_dim} \
               --encoder-attention-heads {encoder_attention_heads} --decoder-attention-heads {decoder_attention_heads} \
               --encoder-normalize-before --decoder-normalize-before \
               --dropout {dropout} --attention-dropout {attention_dropout} --relu-dropout {relu_dropout} \
               --weight-decay {weight_decay} \
               --criterion {criterion} \
               --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 1 \
               --lr-scheduler inverse_sqrt --warmup-updates {warmup_updates} --warmup-init-lr {warmup_init_lr} \
               --lr {lr} --min-lr {min_lr} \
               --max-tokens {max_tokens} \
               --update-freq {update_freq} \
               --max-epoch {max_epoch} --save-interval {save_interval} --save-dir {save_dir} "
        if user_dir is not None:
            cmd += f'--user-dir {user_dir} '
        if restore_file is not None:
           cmd += f"--restore-file {restore_file} "
        if reset:
           cmd += "--reset-optimizer \
                   --reset-lr-scheduler \
                   --reset-dataloader \
                   --reset-meters "
        cmd += additional_cmds
        if logfile is not None:
            import socket, os
            with open(logfile, 'w') as outf:
                print (socket.gethostname(), file=outf)
                print ("pid:", os.getpid(), file=outf)
                print ("screen: %s" % subprocess.check_output('echo $STY', shell=True).decode('utf'), file=outf)
                outf.flush()
            cmd += f"  2>&1 | tee -a {logfile} "
        if GPUs is not None:
            cmd = 'CUDA_VISIBLE_DEVICES={}  {}'.format(GPUs, cmd)
        subprocess.run(cmd, shell=True)


def fairseq_generate(GPUs, preprocess_dir, checkpoint_path, results_path, src, tgt, gen_subset='test', beam=10, nbest=1, max_len_a=1, max_len_b=50, remove_bpe=None, user_dir=None, use_Popen=True, **kwargs):
    additional_cmds = ''.join([f"--{k.replace('_', '-')} {v} " for k, v in kwargs.items() if not isinstance(v, bool)])
    additional_cmds += ''.join([f"--{k.replace('_', '-')} " for k, v in kwargs.items() if isinstance(v, bool) and v])
    cmd = f"fairseq-generate \
            {preprocess_dir} \
        --source-lang {src} --target-lang {tgt} \
        --gen-subset {gen_subset} \
        --path {checkpoint_path} \
        --max-len-a {max_len_a} \
        --max-len-b {max_len_b} \
        --nbest {nbest} \
        --beam {beam} "
    if remove_bpe is not None:
        cmd += f'--remove-bpe {remove_bpe} '
    if user_dir is not None:
        cmd += f'--user-dir {user_dir} '
    cmd += additional_cmds
    if GPUs is not None:
        cmd = 'CUDA_VISIBLE_DEVICES={}  {}'.format(GPUs, cmd)
    with open(results_path, 'w') as f:
        if use_Popen:
            return subprocess.Popen(cmd, shell=True, stdout=f)
        else:
            return subprocess.run(cmd, shell=True, stdout=f)



def parse_fairseq_preds(pred_path):
    #
    preds = defaultdict(list)
    srcs = {}
    with open(pred_path, 'r') as f:
        for line in f:
            if line.startswith('H-'):
                toks = line.split('\t')
                idx = toks[0]
                prob = toks[1]
                pred = ' '.join(toks[2:])
                idx = int(idx[2:])
                preds[idx].append(pred.strip())
            elif line.startswith('S-'):
                toks = line.split('\t')
                idx = toks[0]
                src = ' '.join(toks[1:])
                idx = int(idx[2:])
                srcs[idx] = src.strip()
    # unshuffle & join
    out = []
    for idx in sorted(preds.keys(), key=lambda x: x):
        out.append({'src': srcs[idx], 'pred': preds[idx]})
    return out
