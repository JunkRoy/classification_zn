import os
import pandas as pd
import jieba
from collections import Counter


def preprocess():
    home = 'H:/Python_Workspace/classification_zn/emotion/data'
    file_name = 'waimai_10k.csv'

    datas = pd.read_csv(os.path.join(home, file_name), encoding='utf-8')
    print(datas.keys())
    print(datas['review'][:10])

    lines = []

    for i, d in enumerate(datas['review']):
        print(type(d))
        tokens = jieba.cut(d, cut_all=True)

        line = {
            'text': d,
            'tokens': [t for t in tokens],
            'label': datas['label'][i]
        }
        print(line)
        lines.append(str(line) + '\n')
    save_name = 'corpus.json'

    with open(os.path.join(home, save_name), 'w', encoding='utf-8') as fout:
        fout.writelines(lines)


def statistic(home=None):
    home = 'H:\\Python_Workspace\\classification_zn\\emotion\data'
    file_path = 'corpus.json'

    with open(os.path.join(home, file_path), 'r', encoding='utf-8') as f:
        corpus = []
        length = []
        types = []
        for line in f:
            one = eval(line.strip())
            corpus.append(one)
            length.append(len(one['text']))
            types.append(one['label'])
        print(Counter(length))
        print(Counter(types))


if __name__ == '__main__':
    statistic()
