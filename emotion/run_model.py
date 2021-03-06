import tensorflow as tf
from emotion.utils.utils import creat_bunch, get_vocab, preprocess
from emotion.utils.batch_iter import batch_iter
from tf_models.CNN_moduls import TextCNN
import os


def train():
    # home = ''
    # train_file = ''
    # dev_file = ''
    # # vocab_file = ''
    #
    # vocab_dict = get_vocab()
    # train_bunch = creat_bunch(os.path.join(home, train_file), )


    train_bunch, def_bunch, vocab_dict, word2vec = preprocess()



    batch_train = batch_iter(list(zip(train_bunch.ids, train_bunch.input_x, train_bunch.input_y)))
    with tf.Session() as sess:
        model = TextCNN(vocab_dict, sequence_length=100, num_classes=2, vocab_size=len(vocab_dict),
                        embedding_size=100, filter_sizes=[1, 2, 3, 4, 5], num_filters=128)

        def feed_data(batch):
            feed_dict = {
                model.input_x:batch.input_x,
                model.input_y: batch.input_y,
                model.dropout_keep_prob:0.5
            }
            return feed_dict

        for i, batch in enumerate(batch_train):
            feed_dict = feed_data(batch)
            acc, loss = sess.run([model.accuracy, model.loss], feed_dict=feed_dict)

            print(loss, acc)


if __name__ == '__main__':
    train()
