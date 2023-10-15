from argparse import ArgumentParser


def init_preprocess_options(parser: ArgumentParser):
    parser.add_argument("--data_path", default="D:\\data\\iwslt14.tokenized.de-en")
    parser.add_argument("--src_lang", default="de")
    parser.add_argument("--tgt_lang", default="en")
    parser.add_argument("--dist_dir", default="data")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--vocab_name", default="bpevocab")
    parser.add_argument("--lines_per_thread", type=int, default=1000)

    return parser
