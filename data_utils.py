import torch
import math
import itertools
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class CodonTokenizer:

    def __init__(self, len_u5, len_cds, len_u3):
        bases = ['A', 'C', 'G', 'T']
        codons = [''.join(p) for p in itertools.product(bases, repeat=3)]
        self.mapper = {codon: i + 1 for i, codon in enumerate(codons)}
        self.vocab_size = len(self.mapper) + 1
        self.u5_max = len_u5
        self.cds_max = len_cds
        self.u3_max = len_u3

    def _tokenize(self, seq: str, max_len: int) -> List[int]:
        seq = str(seq).upper().replace('N', 'A')
        tokens = []
        for i in range(0, len(seq), 3):
            codon = seq[i:i + 3]
            if len(codon) == 3:
                tokens.append(self.mapper.get(codon, 1))

        real_len = len(tokens)
        if real_len > max_len:
            tokens = tokens[:max_len]
        else:
            tokens = tokens + [0] * (max_len - real_len)
        return tokens

    def process(self, full_seq: str, u5_len: int, cds_len: int, u3_len: int):
        u5_len, cds_len, u3_len = int(u5_len), int(cds_len), int(u3_len)
        t_u5 = self._tokenize(full_seq[:u5_len], self.u5_max)
        t_cds = self._tokenize(full_seq[u5_len: u5_len + cds_len], self.cds_max)
        t_u3 = self._tokenize(full_seq[u5_len + cds_len:], self.u3_max)

        return (torch.tensor(t_u5, dtype=torch.long),
                torch.tensor(t_cds, dtype=torch.long),
                torch.tensor(t_u3, dtype=torch.long))


class HybridDataset(Dataset):

    def __init__(self, excel_path: str, npy_path: str, tokenizer: CodonTokenizer, name: str = "Dataset"):
        self.df = pd.read_excel(excel_path)
        self.bert_feats = np.load(npy_path)
        self.tokenizer = tokenizer

        self.raw_labels = self.df['mean_te'].values.astype(float)
        self.mean = np.mean(self.raw_labels)
        self.std = np.std(self.raw_labels) + 1e-6
        self.norm_labels = (self.raw_labels - self.mean) / self.std

        logger.info(f"[{name}] Loaded {len(self.raw_labels)} samples.")

        bases = ['A', 'C', 'G', 'T']
        codons = [''.join(p) for p in itertools.product(bases, repeat=3)]
        self.codon_map = {codon: i for i, codon in enumerate(codons)}

        self.token_data = []
        self.bio_feats = []

        for idx, row in self.df.iterrows():
            seq = str(row['tx_sequence']).upper().replace('N', 'A')
            u5, cds, u3 = int(row['utr5_size']), int(row['cds_size']), int(row['utr3_size'])
            self.token_data.append(self.tokenizer.process(seq, u5, cds, u3))

            s_cds = seq[u5: u5 + cds]
            cds_tokens = []
            if len(s_cds) >= 3:
                for i in range(0, len(s_cds), 3):
                    codon = s_cds[i:i + 3]
                    idx_c = self.codon_map.get(codon, 64)
                    if idx_c < 64:
                        cds_tokens.append(idx_c)

            if len(cds_tokens) > 0:
                t = torch.tensor(cds_tokens, dtype=torch.long)
                freq = torch.bincount(t, minlength=65).float() / len(cds_tokens)
                freq = torch.log(freq + 1e-6)
            else:
                freq = torch.zeros(65)

            gc = (s_cds.count('G') + s_cds.count('C')) / (len(s_cds) + 1e-6)
            slen = math.log(len(s_cds) + 1e-6)
            self.bio_feats.append(torch.cat([freq, torch.tensor([gc, slen])], dim=0))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return (*self.token_data[idx],
                torch.tensor(self.bert_feats[idx], dtype=torch.float32),
                self.bio_feats[idx],
                torch.tensor(self.norm_labels[idx], dtype=torch.float32))