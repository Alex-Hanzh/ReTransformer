from argparse import ArgumentParser


def init_preprocess_options(parser: ArgumentParser):
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--src_lang", required=True)
    parser.add_argument("--tgt_lang", required=True)
    parser.add_argument("--dist_dir", default="data-bin")
    parser.add_argument("--workers", default=8)
    parser.add_argument("--vocab_name", default="bpevocab")
    parser.add_argument("--lines-per-thread", default=1000)

    return parser
