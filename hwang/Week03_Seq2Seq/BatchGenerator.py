import random
import numpy as np

class BatchGenerator():
    def __init__(self, in_data: list, out_data: list, batch_size: int):
        self.in_data = in_data # numpy.ndarray
        self.out_data = out_data # numpy.ndarray
        self.batch_size = batch_size

        self.data_size = len(in_data)
        self.epoch = 0
        self.cursor = 0
        self.shuffle_index = list(range(self.data_size))
        random.shuffle(self.shuffle_index)
        self.epoch_end = False

    def next_batch(self):
        # print ("next_batch")
        next_cursor = self.batch_size + self.cursor
        if next_cursor >= self.data_size:
            batch_in_data = [self.in_data[idx] for idx in self.shuffle_index[self.cursor:]]
            batch_out_data = [self.out_data[idx] for idx in self.shuffle_index[self.cursor:]]
            self.epoch += 1
            random.shuffle(self.shuffle_index)
            self.cursor = 0
            self.epoch_end = True
        else:
            batch_in_data = \
                [self.in_data[idx] for idx in self.shuffle_index[self.cursor:next_cursor]]

            batch_out_data = \
                [self.out_data[idx] for idx in self.shuffle_index[self.cursor:next_cursor]]
            self.cursor = next_cursor
            self.epoch_end = False

        return batch_in_data, batch_out_data

    def get_epoch(self):
        return self.epoch

    def get_epoch_end(self):
        return self.epoch_end