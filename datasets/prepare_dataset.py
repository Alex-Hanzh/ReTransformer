from torch.utils.data.dataset import T_co
from torch.utils.data import DataLoader
from utils import batch_by_size, collate_fn
from dataset import (
    WordDict,
    MonolingualDataset,
    LanguagePairDataset,
    LanguagePairIterableDataset,
)
import os
import os.path as path
import pickle


def prepare_monolingual_dataset(data_path, lang, split, target):
    with open(path.join(data_path, f"{split}.{lang}"), "rb") as fl:
        data = pickle.load(fl)

    word_dict = WordDict(path.join(data_path, "dict.txt"))

    return MonolingualDataset(word_dict, data, target)


def prepare_language_pair_dataset(data_path, src_lang, tgt_lang, split):
    src = prepare_monolingual_dataset(data_path, src_lang, split, target="none")
    tgt = prepare_monolingual_dataset(data_path, tgt_lang, split, target="future")

    return LanguagePairDataset(src, tgt)


def prepare_dataloader(
    data_path,
    src_lang,
    tgt_lang,
    split,
    max_tokens,
    strategy="tgt_src",
    long_first=True,
):
    dataset = prepare_language_pair_dataset(data_path, src_lang, tgt_lang, split)
    batch_sampler, info = batch_by_size(
        dataset.src.data,
        dataset.tgt.data,
        max_tokens,
        strategy=strategy,
        long_first=long_first,
    )

    iter_dataset = LanguagePairIterableDataset(dataset, batch_sampler)

    return (
        DataLoader(iter_dataset, None, collate_fn=collate_fn, pin_memory=True),
        dataset.src.word_dict,
    )


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    dataset = prepare_language_pair_dataset(
        "..\\data\\iwslt14.tokenized.de-en", src_lang="de", tgt_lang="en", split="train"
    )
    print(dataset.__getitem__(0))
