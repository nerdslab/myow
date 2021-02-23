import torch
import numpy as np
from torch.utils.data import IterableDataset


class LocalGlobalGenerator(IterableDataset):
    r"""Pair generator for trial-based dataset.

    Args:
        features (numpy.ndarray): Feature matrix of size (num_samples, num_features).
        pair_set (numpy.ndarray): Matrix containing information about possible pairs of features. (N, 3) dimensions
            where the two first columns are the ids for the pair of neighbor features and third column is the id for
            the sequence they belong to.
        sequence_lengths (numpy.ndarray): Number of data points in each sequence. (num_of_sequences)
        batch_size (int): Batch size.
        pool_batch_size (int): Batch size of candidate samples.
        num_examples (int): Total number of examples.
        transform (Callable, Optional): Transformation.
        structure_transform (bool, Optional): If :obj:`True`, the same transformation to generate augmented views.
            (default: :obj:`False`)
        num_workers (int, Optional): Number of workers used in dataloader. (default: :obj:`1`)
    """
    def __init__(self, features, pair_set, sequence_lengths, batch_size, pool_batch_size, num_examples,
                 transform=None, structured_transform=False, num_workers=1):
        self.features = features
        self.pair_set = pair_set[:, 1:]
        self.sequence_ids = pair_set[:, 0]

        sequence = []
        for i, n in enumerate(sequence_lengths):
            sequence.append(np.ones(n) * i)
        self.sequence = np.concatenate(sequence).astype(int)

        self.num_trials = int(self.sequence_ids[-1]) + 1
        self.batch_size = batch_size
        self.pool_batch_size = pool_batch_size
        self.transform = transform
        self.structured_transform = structured_transform
        self.sequence_mask, self.random_mask = self._build_masks()
        self.num_workers = num_workers
        self.num_iterations = int(num_examples // batch_size)

    def _build_masks(self):
        pair_masks = {}
        random_masks = {}
        for i in range(self.num_trials):
            pair_masks[i] = self.sequence_ids == i
            random_masks[i] = self.sequence == i
        return pair_masks, random_masks

    def _sample_trials(self):
        permuted_trials = np.random.permutation(self.num_trials)
        split_trials = int(self.num_trials / 2 + np.random.normal(self.num_trials / 10))
        trials_1 = permuted_trials[:split_trials]
        trials_2 = permuted_trials[split_trials:]
        return trials_1, trials_2

    def _sample_pairs(self):
        sample_indices = self.pair_set[np.random.choice(self.pair_set.shape[0], self.batch_size, replace=False)].T
        x1, x2 = self.features[sample_indices, :]
        dt = sample_indices[1] - sample_indices[0]
        return x1, x2, dt

    def _sample_random(self, from_trials, batch_size):
        mask = np.sum([self.random_mask[i] for i in from_trials], axis=0).astype(bool)
        if sum(mask) >= batch_size:
            sample_indices = np.random.choice(sum(mask), batch_size, replace=False)
        else:
            sample_indices = np.concatenate([np.arange(sum(mask)),
                                             np.random.choice(sum(mask), batch_size-sum(mask), replace=False)])
        x = self.features[mask][sample_indices]
        return x

    def __iter__(self):
        for _ in range(self.num_iterations):
            trials_1, trials_2 = self._sample_trials()
            x1, x2, dt = self._sample_pairs()
            x3 = self._sample_random(trials_1, self.batch_size)
            x4 = self._sample_random(trials_2, self.pool_batch_size)

            x1 = torch.tensor(x1)
            x2 = torch.tensor(x2)
            x3 = torch.tensor(x3)
            x4 = torch.tensor(x4)

            if self.transform is not None:
                if self.structured_transform:
                    x1, x2 = self.transform(x1, x2)
                else:
                    [x1] = self.transform(x1)
                    [x2] = self.transform(x2)
                [x3] = self.transform(x3)
                [x4] = self.transform(x4)
            yield x1.type(torch.float32), x2.type(torch.float32), \
                  x3.type(torch.float32), x4.type(torch.float32)

    def __len__(self):
        return self.num_iterations * self.batch_size * self.num_workers

    @staticmethod
    def prepare_views(inputs):
        x1, x2, x3, x4 = inputs
        outputs = {'view1': x1.squeeze(), 'view2': x2.squeeze(),
                   'view3': x3.squeeze(), 'view_pool': x4.squeeze()}
        return outputs
