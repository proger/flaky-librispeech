#!/usr/bin/env python

import argparse
import pandas as pd
import numpy as np
import sentencepiece as spm
import sys

sp = spm.SentencePieceProcessor(model_file='libribpe.model')

parser = argparse.ArgumentParser(description='simulate uniform cleanups for active learning')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--topk', type=int, help='generate this many chunks of mixtures')
args = parser.parse_args()

ref = pd.read_csv('data/flaky/train-clean-100.ref.txt', sep='\t')
ep45 = pd.read_csv('data/flaky/train-clean-100.ep45.txt', sep='\t')

both = pd.concat([ep45, ref], axis=1)
both = both.sample(frac=1, replace=False, random_state=args.seed)
both.columns = ['filename', 'dirty', 'filename1', 'clean']

chunk_size = 2196

chunks = [both[i:i+chunk_size] for i in range(0,len(both),chunk_size)]
topk = args.topk if args.topk else len(chunks)

def encode(x):
    return ' '.join(sp.encode(x, out_type=str))

for i in range(0,len(chunks)+1)[::-1][:topk]:
    query = chunks[i-1]
    clean = pd.concat(chunks[:i]) if chunks[:i] else both.sample(0)
    dirty = pd.concat(chunks[i:]) if chunks[i:] else both.sample(0)

    query = query[['filename', 'clean']]
    query.columns = ['filename', 'text']

    known_clean = clean[['filename', 'clean']]
    known_clean.columns = ['filename', 'text']

    known_dirty = clean[['filename', 'dirty']][clean['dirty'] != clean['clean']]
    known_dirty.columns = ['filename', 'text']

    unknown = dirty[['filename', 'dirty']]
    unknown.columns = ['filename', 'text']

    #if len(unknown) == 24146:
    #    import ipdb; ipdb.set_trace()

    test_unk = unknown.copy()

    query['text'] = '<↑> ' + query['text'].apply(encode)
    known_clean['text'] = '<↑> ' + known_clean['text'].apply(encode)
    known_dirty['text'] = '<↓> ' + known_dirty['text'].apply(encode)
    unknown['text'] = '<?> ' + unknown['text'].apply(encode)
    test_unk['text'] = test_unk['text'].apply(encode)

    counts = np.array([len(known_clean), len(known_dirty), len(unknown)])
    if np.any(counts == 0):
        print('could not produce oversampled dataset for chunk', i, counts)
        continue
    
    total = np.sum(counts)
    frac = np.where(counts, total/(3*counts), 0)
    frac /= np.min(frac)

    dataset = pd.concat([
        known_clean.sample(frac=frac[0], replace=True),
        known_dirty.sample(frac=frac[1], replace=True),
        unknown.sample(frac=frac[2], replace=True)
    ])
    dataset = dataset.sample(frac=1, replace=False, random_state=i)

    filename = f'data/flaky/spin_oversampled/train-clean-100.seed{args.seed}.unknown{len(unknown):05d}.unk.txt.spin.test'
    test_unk.to_csv(filename, sep='\t', index=False, header=False)

    filename = f'data/flaky/spin_oversampled/train-clean-100.seed{args.seed}.unknown{len(unknown):05d}.txt.spin'
    dataset.to_csv(filename, sep='\t', index=False, header=False)

    filename = f'data/flaky/spin_oversampled/train-clean-100.seed{args.seed}.unknown{len(unknown):05d}.clean.txt.spin'
    clean.to_csv(filename, sep='\t', index=False, header=False)

    filename = f'data/flaky/spin_oversampled/train-clean-100.seed{args.seed}.unknown{len(unknown):05d}.query.txt.spin'
    query.to_csv(filename, sep='\t', index=False, header=False)

print([len(chunk) for chunk in chunks], len(chunks))
