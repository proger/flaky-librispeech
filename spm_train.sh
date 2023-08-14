#curl https://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz | gunzip -c > data/corrupted-librispeech/librispeech-lm-norm.txt
find data/LibriSpeech/train-* | grep txt  | xargs cat | cut -d' ' -f2- > data/corrupted-librispeech/train.txt
spm_train --bos_id 2 --eos_id 3 --pad_id 0 --unk_id 1 --vocab_size 512 --character_coverage 1.0 --input data/corrupted-librispeech/train.txt --model_prefix exp/libri
spm_train --bos_id 2 --eos_id 3 --pad_id 0 --unk_id 1 --vocab_size 512 --character_coverage 1.0 --input data/corrupted-librispeech/train.txt --model_prefix exp/libribpe --model_type bpe
spm_train --bos_id 2 --eos_id 3 --pad_id 0 --unk_id 1 --vocab_size 128 --character_coverage 1.0 --input data/corrupted-librispeech/train.txt --model_prefix exp/libribpe128 --model_type bpe
