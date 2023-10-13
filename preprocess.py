from argparse import ArgumentParser
from multiprocessing import Pool
from Transformer.handle import init_preprocess_options
import os
import pickle


def handle_args(args):
    """handle script task based on parameters

    Args:
        args : command line arguments
    """
    word2idx = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "unk": 3}


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = init_preprocess_options(parser)
    args = parser.parse_args()
    handle_args(args)
