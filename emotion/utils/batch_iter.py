import numpy as np

def batch_iter(data, batch_size=64, num_epoch=5, shuffle=True):
    data = list(data)
    data_size = len(data)

    num_batches_per_epoch = int(len(data)/batch_size)

    for epoch in range(num_epoch):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            data = np.array(data, dtype=object)
            shuffle_data = data[shuffle_indices]
        else:
            shuffle_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num+1)*batch_size, data_size)
            yield shuffle_data[start_index:end_index]