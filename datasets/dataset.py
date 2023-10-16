import torch
from copy import deepcopy
from torch.utils.data import Dataset, RandomSampler, IterableDataset
from torch.utils.data.dataset import T_co
from torch.utils.data import Dataset
from typing import Iterator


class WordDict(object):
    """
    WordDict 构造字典类，实现word2idx idx2word tokenize detokenize功能
    """

    def __init__(self, word_dict_path) -> None:
        super().__init__()

        with open(word_dict_path, "r", encoding="utf-8") as f:
            self.vocab_size = int(f.readline().strip())
            self.word_dict = {}
            self.idx_dict = {}
            for line in f.readlines():
                word, ind = line.strip().split()
                ind = int(ind)
                self.word_dict[word] = ind
                self.idx_dict[ind] = word
                if word[0] == "<":
                    sp_tokens = word[1:-1]
                    if sp_tokens == "pad":
                        self.padding_idx = ind
                    elif sp_tokens == "unk":
                        self.unknown_idx = ind
                    elif sp_tokens == "bos":
                        self.bos_idx = ind
                    elif sp_tokens == "eos":
                        self.eos_idx = ind

    def idx2word(self, idx: int):
        return self.idx_dict[idx]

    def word2idx(self, word: str):
        return self.word_dict[word]

    def tokenize(self, sentence: str):
        sentence = [self.word_dict[word] for word in sentence.strip().split()]
        return sentence

    def detokenize(self, tensor: torch.Tensor):
        if isinstance(tensor, torch.Tensor):
            if len(tensor.shape) == 2:
                sentence = [
                    " ".join([self.idx_dict[idx] for idx in sent])
                    for sent in tensor.cpu().tolist()
                ]
            elif len(tensor.shape) == 1:
                sentence = " ".join(
                    [self.idx_dict[idx] for idx in tensor.cpu().tolist()]
                )
            else:
                raise Exception()
        else:
            sentence = " ".join([self.idx_dict[idx] for idx in tensor])

        return sentence


class MonolingualDataset(Dataset):
    def __init__(self, word_dict: WordDict, data: list, target="future") -> None:
        super().__init__()

        self.data = data
        self.word_dict = word_dict

        assert target in ["future", "past", "present", "none"]

        self.target = target

    def __getitem__(self, index) -> T_co:
        source = self.data[index][:]
        target = self.data[index][:]

        source = [self.word_dict.bos_idx] + source

        if self.target == "past":
            target = [self.word_dict.bos_idx, self.word_dict.bos_idx] + target
            source = source + [self.word_dict.eos_idx]
        elif self.target == "future":
            target = target + [self.word_dict.eos_idx]
        elif self.target == "present":
            target = [self.word_dict.bos_idx] + target + [self.word_dict.eos_idx]
            source = source + [self.word_dict.eos_idx]
        elif self.target == "none":
            target = None
        else:
            raise Exception(f"Target type {self.target} is not supported!")

        return {"id": index, "data": {"source": source, "target": target}}

    def __len__(self):
        return len(self.data)


class LanguagePairDataset(Dataset):
    def __init__(self, src: MonolingualDataset, tgt: MonolingualDataset) -> None:
        super().__init__()
        assert len(src) == len(tgt)
        self.src = src
        self.tgt = tgt

    def __getitem__(self, index) -> T_co:
        return {
            "id": index,
            "data": {
                "src_lang": self.src[index]["data"]["source"],
                "tgt_lang": self.tgt[index]["data"]["source"],
                "target": self.tgt[index]["data"]["target"],
            },
        }

    def __len__(self):
        return len(self.src)


class LanguagePairIterableDataset(IterableDataset):
    def __init__(self, dataset: LanguagePairDataset, batch_sampler) -> None:
        super().__init__()
        self.dataset = dataset
        self.batch_sampler = batch_sampler

        self.padding()

    def padding(self):
        batch_sampler = []
        for batch in self.batch_sampler:
            sampler = batch
            batch = []
            max_tgt_len = 0
            max_src_len = 0
            for sample in sampler:
                data = self.dataset[sample]
                max_src_len = max(max_src_len, len(data["data"]["src_lang"]))
                max_tgt_len = max(max_tgt_len, len(data["data"]["tgt_lang"]))
            for sample in sampler:
                data = deepcopy(self.dataset[sample])
                data["data"]["src_lang"] = [
                    self.dataset.src.word_dict.padding_idx
                    for _ in range(max_src_len - len(data["data"]["src_lang"]))
                ] + data["data"]["src_lang"]
                data["data"]["tgt_lang"] += [
                    self.dataset.tgt.word_dict.padding_idx
                    for _ in range(max_tgt_len - len(data["data"]["tgt_lang"]))
                ]
                data["data"]["target"] += [
                    self.dataset.tgt.word_dict.padding_idx
                    for _ in range(max_tgt_len - len(data["data"]["target"]))
                ]
                batch.append(data)
            batch_sampler.append(batch)
        self.batch_sampler = batch_sampler

    def __iter__(self) -> Iterator[T_co]:
        batch_sampler = []
        for batch in self.batch_sampler:
            batch_sampler.append([batch[i] for i in RandomSampler(batch)])
        sample_ind = RandomSampler(batch_sampler)
        for ind in sample_ind:
            yield batch_sampler[ind]

    def __len__(self):
        return len(self.batch_sampler)
