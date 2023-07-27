#!/usr/bin/env python

import argparse
import pandas as pd
import itertools

parser = argparse.ArgumentParser(description='simulate one random chunk cleanup for active learning')
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()  

ref = pd.read_csv('data/corrupted-librispeech/train-clean-100.ref.txt', sep='\t')
ep45 = pd.read_csv('data/corrupted-librispeech/train-clean-100.ep45.txt', sep='\t')

both = pd.concat([ep45, ref], axis=1)
both = both.sample(frac=1, replace=False, random_state=args.seed)
both.columns = ['filename', 'dirty', 'filename1', 'clean']

chunk_size = 2196

chunks = [both[i:i+chunk_size] for i in range(0,len(both),chunk_size)]

for i in range(0,len(chunks)):
    clean = chunks[i]
    dirty = pd.concat(chunks[:i] + chunks[i+1:])

    clean = clean[['filename', 'clean']]
    clean.columns = ['filename', 'text']

    dirty = dirty[['filename', 'dirty']]
    dirty.columns = ['filename', 'text']

    dataset = pd.concat([clean, dirty])
    dataset = dataset.sample(frac=1, replace=False, random_state=i)

    filename = f'data/corrupted-librispeech/train-clean-100.seed{args.seed}.dirty{len(dirty):05d}.clean{i:02d}.txt'
    print(filename, i, len(dirty), len(clean), len(dataset))
    dataset.to_csv(filename, sep='\t', index=False, header=False)
    
print([len(chunk) for chunk in chunks], len(chunks), sum([len(chunk) for chunk in chunks]))
