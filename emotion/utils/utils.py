import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=Warning)

import numpy as np
import tensorflow as tf
import yaml
from sklearn.datasets.base import Bunch
import os
import pickle


## TODO 改config

def config_init(flags, config_path):
    with open(corpus_path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
        flags.DEFINE_integer("emb_size", cfg['hyperparameters']['emb_size'])


def get_vocab(config=None, word2vec_path=None, corpus_path=None):
    emb_path = '../model/vocab/vocab.txt'
    vocab = {}
    if os.path.exists(emb_path):
        with open(emb_path, 'r', encoding='utf-8') as f:
            for line in f:
                one = line.strip().split('\t')
                vocab[one[0]] = one[1]
    else:
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                one = eval(line.strip())
                for word in one['tokens']:
                    if word not in vocab.keys():
                        vocab[word] = len(vocab.keys()) + 1
        vocab['UNK'] = 0
        lines = [str(key) + '\t' + str(value) + '\n' for key, value in vocab.items()]

        with open(emb_path, 'w', encoding='utf-8') as fout:
            fout.writelines(lines)
    return vocab


def get_inputs(sample, vocab_dict, config=None):
    corpus_path = ''

    max_seq_length = 100
    tokens = sample['tokens']
    label = sample['label']
    x = [vocab_dict.get(token, 'UNK') for token in tokens]
    y = label
    if len(x) > max_seq_length:
        x = x[:100]
    else:
        x += [0] * (100 - len(x))
    return np.array(x), np.array(y)


def creat_bunch(path, vocab_dict):

    bunch = Bunch(ids=[], input_x=[], input_y=[])

    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            sample = eval(line.strip())
            x, y = get_inputs(sample, vocab_dict)
            bunch.ids.append('train_' + str(i))
            bunch.input_x.append(x)
            bunch.input_y.append(y)
    return bunch


def preprocess(config=None):
    # corpus_path = '../data/corpus.json'
    train_path = '../data/train.json'
    dev_path = '../data/dev.json'
    vocab_dict = get_vocab(config=None)
    train_bunch = creat_bunch(train_path, vocab_dict)
    dev_bunch = creat_bunch(dev_path, vocab_dict)

    return train_bunch, dev_bunch, vocab_dict


if __name__ == '__main__':
    corpus_path = 'D:\\PythonWorkspace\\GitWorkspace\\classification_zn\\' \
                  'emotion\\data\\corpus.json'

    get_vocab(config=None, corpus_path=corpus_path)
