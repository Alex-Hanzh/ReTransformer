# -*- coding: utf-8 -*-
import os

os.chdir(os.path.dirname(__file__))

from datasets import *
import yaml

for i, batch in enumerate(train_iter):
    src = batch.src
    print(src[0])
    input()
