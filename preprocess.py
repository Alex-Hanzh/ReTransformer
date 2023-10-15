from argparse import ArgumentParser
from math import pi
from multiprocessing import Pool
from telnetlib import PRAGMA_HEARTBEAT

from requests import patch
from config.args_options import init_preprocess_options
import os
import pickle


def handle_data_multi_process(
    src_lines: list, tgt_lines: list, word2idx: dict, is_test: bool
):
    """
    handle_data_multi_process

    Args:
        src_lines (string): source language lines
        tgt_lines (string): target language lines
        word2idx (dict): map of word to index
        is_test (bool): is test or not

    Returns:
        src_res,tgt_res [List,List]: idx list of src_lines & tgt_lines
    """

    src_res = []
    tgt_res = []

    for src_line, tgt_line in zip(src_lines, tgt_lines):
        src_res.append(
            [
                word2idx[i] if i in word2idx else word2idx["<unk>"]
                for i in src_line.strip().split(" ")
            ]
        )

        if is_test:  # spilt the token list & copy to res
            tgt_res.append([i for i in tgt_line.strip().split(" ")])
        else:
            tgt_res.append(
                [
                    word2idx[i] if i in word2idx else word2idx["<unk>"]
                    for i in tgt_line.strip().split(" ")
                ]
            )

    return src_res, tgt_res


def handle_args(args):
    """handle script task based on parameters

    Args:
        args : command line arguments
    """
    word2idx = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}  # unk for unkown tokens
    root_dir: str = args.data_path
    if root_dir[-1] in ["/", "\\"]:
        root_dir = root_dir[:-1]

    dist_dir = os.path.join(args.dist_dir, os.path.basename(root_dir))

    with open(os.path.join(root_dir, args.vocab_name), "r", encoding="utf8") as vocab:
        vocab_size = 0
        for idx, line in enumerate(vocab.readlines()):
            line = line.strip().split()
            word2idx[line[0]] = idx + 4
            vocab_size = idx + 5

    if not os.path.exists(dist_dir):
        os.makedirs(dist_dir)

    with open(os.path.join(dist_dir, "dict.txt"), "w", encoding="utf8") as fl:
        print(vocab_size, file=fl)
        for k, v in word2idx.items():
            print(f"{k} {v}", file=fl)

    for split in ["train", "test", "valid"]:
        pool = Pool(args.workers)
        with open(
            os.path.join(root_dir, f"{split}.{args.src_lang}"), "r", encoding="utf-8"
        ) as src, open(
            os.path.join(root_dir, f"{split}.{args.tgt_lang}"), "r", encoding="utf-8"
        ) as tgt:
            src_res = []
            tgt_res = []

            result = []

            src_lines = []
            tgt_lines = []

            for idx, (src_line, tgt_line) in enumerate(
                zip(src.readlines(), tgt.readlines())
            ):
                if idx > 0 and idx % args.lines_per_thread == 0:
                    result.append(
                        pool.apply_async(
                            handle_data_multi_process,
                            (src_lines, tgt_lines, word2idx, split == "test"),
                        )
                    )
                    src_lines = []
                    tgt_lines = []

                src_lines.append(src_line)
                tgt_lines.append(tgt_line)

            if len(src_lines):
                result.append(
                    pool.apply_async(
                        handle_data_multi_process,
                        (src_lines, tgt_lines, word2idx, split == "test"),
                    )
                )
            pool.close()
            pool.join()
            for res in result:
                res = res.get()
                src_res += res[0]
                tgt_res += res[1]

        with open(
            os.path.join(dist_dir, f"{split}.{args.src_lang}"), "wb"
        ) as src, open(os.path.join(dist_dir, f"{split}.{args.tgt_lang}"), "wb") as tgt:
            pickle.dump(src_res, src)
            pickle.dump(tgt_res, tgt)


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    parser = ArgumentParser()
    parser = init_preprocess_options(parser)
    args = parser.parse_args()
    handle_args(args)
