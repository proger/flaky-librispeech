import os

import torch
import torchaudio


def make_frames(wav, op='fbank'):
    if op == 'mfcc':
        frames = torchaudio.compliance.kaldi.mfcc(wav)

        # utterance-level CMVN
        frames -= frames.mean(dim=0)
        frames /= frames.std(dim=0)
    elif op == 'fbank':
        frames = torchaudio.compliance.kaldi.fbank(wav, num_mel_bins=80)
    
        # whisper-style normalization
        frames += 8.
        frames /= 4.
    else:
        assert False

    return frames # (T, F)


class LabelFile(torch.utils.data.Dataset):
    def __init__(self, path: os.PathLike):
        super().__init__()
        with open(path) as f:
            self.ark = dict(line.strip().split(maxsplit=1) for line in f)
            self.filenames = list(self.ark.keys())

    def __len__(self):
        return len(self.filenames)

    def utt_id(self, index):
        return self.filenames[index]

    def __getitem__(self, index):
        wav, sr = torchaudio.load(self.filenames[index])
        assert sr == 16000
        return index, make_frames(wav), self.ark[self.filenames[index]]


class LibriSpeech(torch.utils.data.Dataset):
    def __init__(self, url='train-clean-100'):
        super().__init__()
        self.librispeech = torchaudio.datasets.LIBRISPEECH('data', url=url, download=True)

    def __len__(self):
        return len(self.librispeech)

    def utt_id(self, index):
        wav, sr, text, speaker_id, chapter_id, utterance_id = self.librispeech[index]
        utt_id = f'{speaker_id}-{chapter_id}-{utterance_id:04d}'
        return utt_id

    def __getitem__(self, index):
        wav, sr, text, speaker_id, chapter_id, utterance_id = self.librispeech[index]
        return index, make_frames(wav), text


if __name__ == '__main__':
    for (i, f, t), (i2, f2, t2) in zip(LibriSpeech(), LabelFile("train-clean-100.ep45.txt")):
        print(i, t[:40], t2[:40], sep='\t')
