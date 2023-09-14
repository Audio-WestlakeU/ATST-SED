from torch.utils.data import Sampler
import numpy as np


class ConcatDatasetBatchSampler(Sampler):
    """This sampler is built to work with a standard Pytorch ConcatDataset.
    From SpeechBrain dataio see https://github.com/speechbrain/

    It is used to retrieve elements from the different concatenated datasets placing them in the same batch
    with proportion specified by batch_sizes, e.g 8, 16 means each batch will
    be of 24 elements with the first 8 belonging to the first dataset in ConcatDataset
    object and the last 16 to the second.
    More than two datasets are supported, in that case you need to provide 3 batch
    sizes.

    Note
    ----
    Batched are drawn from the datasets till the one with smallest length is exhausted.
    Thus number of examples in your training epoch is dictated by the dataset
    whose length is the smallest.


    Arguments
    ---------
    samplers : int
        The base seed to use for the random number generator. It is recommended
        to use a value which has a good mix of 0 and 1 bits.
    batch_sizes: list
        Batch sizes.
    epoch : int
        The epoch to start at.
    """

    def __init__(self, samplers, batch_sizes: (tuple, list), epoch=0) -> None:

        if not isinstance(samplers, (list, tuple)):
            raise ValueError(
                "samplers should be a list or tuple of Pytorch Samplers, "
                "but got samplers={}".format(batch_sizes)
            )

        if not isinstance(batch_sizes, (list, tuple)):
            raise ValueError(
                "batch_sizes should be a list or tuple of integers, "
                "but got batch_sizes={}".format(batch_sizes)
            )

        if not len(batch_sizes) == len(samplers):
            raise ValueError("batch_sizes and samplers should be have same length")

        self.batch_sizes = batch_sizes
        self.samplers = samplers
        self.offsets = [0] + np.cumsum([len(x) for x in self.samplers]).tolist()[:-1]

        self.epoch = epoch
        self.set_epoch(self.epoch)

    def _iter_one_dataset(self, c_batch_size, c_sampler, c_offset):
        batch = []
        for idx in c_sampler:
            batch.append(c_offset + idx)
            if len(batch) == c_batch_size:
                yield batch

    def set_epoch(self, epoch):
        if hasattr(self.samplers[0], "epoch"):
            for s in self.samplers:
                s.set_epoch(epoch)

    def __iter__(self):

        iterators = [iter(i) for i in self.samplers]
        tot_batch = []

        for b_num in range(len(self)):
            for samp_idx in range(len(self.samplers)):
                c_batch = []
                while len(c_batch) < self.batch_sizes[samp_idx]:
                    c_batch.append(self.offsets[samp_idx] + next(iterators[samp_idx]))
                tot_batch.extend(c_batch)
            yield tot_batch
            tot_batch = []

    def __len__(self):

        min_len = float("inf")
        for idx, sampler in enumerate(self.samplers):
            c_len = (len(sampler)) // self.batch_sizes[idx]

            min_len = min(c_len, min_len)
        return min_len
