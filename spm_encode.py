import sentencepiece as spm
import sys

sp = spm.SentencePieceProcessor(model_file='exp/libribpe.model')

for line in sys.stdin:
    key, text = line.split(maxsplit=1)
    print(key, ' '.join(sp.encode(text, out_type=str)))
