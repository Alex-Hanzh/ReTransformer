# ReTransformer

redo the experiment of "attention is all you need"

本项目学习[reference](https://zhuanlan.zhihu.com/p/438123116)的实现方法

## 1. Target & Method

由于是复现Transformer这篇文章，所以第一阶段是复现文章中的翻译任务，数据集采用IWSLT(German to English)，后续扩充WMT(English to German)的实验
方法完全复现Transformer的基本架构，后续加入一些提高性能的trick

## 2. Datasets process

IWSLT数据集已经经过了bpe分词，所以只需要完成：word2index的映射，并将数据集中的sentence的word list转为idx list
这里采用multiprocessing的Pool进程池并行处理数据，能有效的提高CPU效率（btw，也可以使用线程池来实现，但线程同步、GIL锁等问题让其更复杂）

## 3. Datasets & Dataloader

