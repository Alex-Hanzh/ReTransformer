# -*- coding: utf-8 -*-

import argparse

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("square", help="display a square of a given number", type=int)
args = parser.parse_args()
print(args.square**2) 