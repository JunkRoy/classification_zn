import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=Warning)

import numpy as np
import tensorflow as tf
import yaml
import os

#TODO 改config

def config_init(flags, config_path):
    with open(corpus_path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
        flags.DEFINE_integer("emb_size", cfg['hyperparameters']['emb_size'])


def word2vec(config, word2vec_path=None, corpus_path=None):
    emb_path = '../model/vocab/vocab.txt'
    vocab = {}
    if os.path.exists(emb_path):
        with open(emb_path, 'r', encoding='utf-8') as f:
            for line in f:
                one = line.strip().split('\t')
                vocab[one[0]] = one[1]

    if word2vec_path is None:
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

def get_inputs(sample, vocab_dict, config=None):
    corpus_path = ''
    max_seq_length = 100
    tokens = sample['tokens']
    label = sample['label']
    x = [vocab_dict.get(token, 'UNK') for token in tokens]
    y = label
    if len(x)>max_seq_length:
        x = x[:100]
    else:
        x += [0]*(100-len(x))
    return np.array(x), np.array(y)








if __name__ == '__main__':
    corpus_path = 'D:\\PythonWorkspace\\GitWorkspace\\classification_zn\\' \
                  'emotion\\data\\corpus.json'

    word2vec(None, corpus_path=corpus_path)
