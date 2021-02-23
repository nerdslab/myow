import os
import pickle

import numpy as np
from tqdm import tqdm
import torch

from self_supervised.data.io import loadmat


FILENAMES = {
    ('mihi', 1): 'full-mihi-03032014',
    ('mihi', 2): 'full-mihi-03062014',
    ('chewie', 1): 'full-chewie-10032013',
    ('chewie', 2): 'full-chewie-12192013',
}


class ReachNeuralDataset:
    def __init__(self, path, primate='mihi', day=1,
                 binning_period=0.1, binning_overlap=0.0, train_split=0.8,
                 scale_firing_rates=False, scale_velocity=False, sort_by_reach=True):

        self.path = path
        # get path to data
        assert primate in ['mihi', 'chewie']
        assert day in [1, 2]
        self.primate = primate

        self.filename = FILENAMES[(self.primate, day)]
        self.raw_path = os.path.join(self.path, 'raw/%s.mat') % self.filename
        self.processed_path = os.path.join(self.path, 'processed/%s.pkl') % (self.filename + '-%.2f' % binning_period)

        # get binning parameters
        self.binning_period = binning_period
        self.binning_overlap = binning_overlap
        if self.binning_overlap != 0:
            raise NotImplemented

        # train/val split
        self.train_split = train_split

        # initialize some parameters
        self.dataset_ = {}
        self.subset = 'train'  # default selected subset

        ### Process data
        # load data
        if not os.path.exists(self.processed_path):
            data_train_test = self._process_data()
        else:
            data_train_test = self._load_processed_data()

        # split data
        data_train, data_test = self._split_data(data_train_test)
        self._num_trials = {'train': len(data_train['firing_rates']),
                            'test': len(data_test['firing_rates'])}

        # compute mean and std of firing rates
        self.mean, self.std = self._compute_mean_std(data_train, feature='firing_rates')

        # remove neurons with no variance
        data_train, data_test = self._remove_static_neurons(data_train, data_test)

        # scale data
        if scale_firing_rates:
            data_train, data_test = self._scale_data(data_train, data_test, feature='firing_rates')
        if scale_velocity:
            data_train, data_test = self._scale_data(data_train, data_test, feature='velocity')

        # sort by reach direction
        if sort_by_reach:
            data_train = self._sort_by_reach_direction(data_train)
            data_test = self._sort_by_reach_direction(data_test)

        # build sequences
        trial_lengths_train = [seq.shape[0] for seq in data_train['firing_rates']]

        # merge everything
        for feature in data_train.keys():
            data_train[feature] = np.concatenate(data_train[feature]).squeeze()
            data_test[feature] = np.concatenate(data_test[feature]).squeeze()

        data_train['trial_lengths'] = trial_lengths_train
        data_train['reach_directions'] = np.unique(data_train['labels']).tolist()
        data_train['reach_lengths'] = [np.sum(data_train['labels'] == reach_id)
                                       for reach_id in data_train['reach_directions']]

        # map labels to 0 .. N-1 for training
        data_train['raw_labels'] = data_train['labels'].copy()
        data_test['raw_labels'] = data_test['labels'].copy()

        data_train['labels'] = self._map_labels(data_train)
        data_test['labels'] = self._map_labels(data_test)

        self.dataset_['train'] = data_train
        self.dataset_['test'] = data_test

    @property
    def dataset(self):
        return self.dataset_[self.subset]

    def __getattr__(self, item):
        return self.dataset[item]

    def train(self):
        self.subset = 'train'

    def test(self):
        self.subset = 'test'

    @property
    def num_trials(self):
        return self._num_trials[self.subset]

    @property
    def num_neurons(self):
        return self[0]['firing_rates'].shape[1]

    def _process_data(self):
        print('Preparing dataset: Binning data.')
        # load data
        mat_dict = loadmat(self.raw_path)

        # bin data
        data = self._bin_data(mat_dict)

        self._save_processed_data(data)
        return data

    def _save_processed_data(self, data):
        with open(self.processed_path, 'wb') as output:
            pickle.dump({'data': data}, output)

    def _load_processed_data(self):
        with open(self.processed_path, "rb") as fp:
            data = pickle.load(fp)['data']
        return data

    def _bin_data(self, mat_dict):
        # load matrix
        trialtable = mat_dict['trial_table']
        neurons = mat_dict['out_struct']['units']
        pos = np.array(mat_dict['out_struct']['pos'])
        vel = np.array(mat_dict['out_struct']['vel'])
        acc = np.array(mat_dict['out_struct']['acc'])
        force = np.array(mat_dict['out_struct']['force'])
        time = vel[:, 0]

        num_neurons = len(neurons)
        num_trials = trialtable.shape[0]

        data = {'firing_rates': [], 'position': [], 'velocity': [], 'acceleration': [],
                'force': [], 'labels': [], 'sequence': []}
        for trial_id in tqdm(range(num_trials)):
            min_T = trialtable[trial_id, 9]
            max_T = trialtable[trial_id, 12]

            # grids= minT:(delT-TO):(maxT-delT);
            grid = np.arange(min_T, max_T + self.binning_period, self.binning_period)
            grids = grid[:-1]
            gride = grid[1:]
            num_bins = len(grids)

            neurons_binned = np.zeros((num_bins, num_neurons))
            pos_binned = np.zeros((num_bins, 2))
            vel_binned = np.zeros((num_bins, 2))
            acc_binned = np.zeros((num_bins, 2))
            force_binned = np.zeros((num_bins, 2))
            targets_binned = np.zeros((num_bins, 1))
            id_binned = trial_id * np.ones((num_bins, 1))

            for k in range(num_bins):
                bin_mask = (time >= grids[k]) & (time <= gride[k])
                if len(pos) > 0:
                    pos_binned[k, :] = np.mean(pos[bin_mask, 1:], axis=0)
                vel_binned[k, :] = np.mean(vel[bin_mask, 1:], axis=0)
                if len(acc):
                    acc_binned[k, :] = np.mean(acc[bin_mask, 1:], axis=0)
                if len(force) > 0:
                    force_binned[k, :] = np.mean(force[bin_mask, 1:], axis=0)
                targets_binned[k, 0] = trialtable[trial_id, 1]

            for i in range(num_neurons):
                for k in range(num_bins):
                    spike_times = neurons[i]['ts']
                    bin_mask = (spike_times >= grids[k]) & (spike_times <= gride[k])
                    neurons_binned[k, i] = np.sum(bin_mask) / self.binning_period

            data['firing_rates'].append(neurons_binned)
            data['position'].append(pos_binned)
            data['velocity'].append(vel_binned)
            data['acceleration'].append(acc_binned)
            data['force'].append(force_binned)
            data['labels'].append(targets_binned)
            data['sequence'].append(id_binned)
        return data

    def _split_data(self, data):
        num_trials = len(data['firing_rates'])
        split_id = int(num_trials * self.train_split)

        data_train = {}
        data_test = {}
        for key, feature in data.items():
            data_train[key] = feature[:split_id]
            data_test[key] = feature[split_id:]
        return data_train, data_test

    def _remove_static_neurons(self, data_train, data_test):
        for i in range(len(data_train['firing_rates'])):
            data_train['firing_rates'][i] = data_train['firing_rates'][i][:, self.std > 1e-3]
        for i in range(len(data_test['firing_rates'])):
            data_test['firing_rates'][i] = data_test['firing_rates'][i][:, self.std > 1e-3]
        self.mean = self.mean[self.std > 1e-3]
        self.std = self.std[self.std > 1e-3]
        return data_train, data_test

    def _compute_mean_std(self, data, feature='firing_rates'):
        concatenated_data = np.concatenate(data[feature])
        mean = concatenated_data.mean(axis=0)
        std = concatenated_data.std(axis=0)
        return mean, std

    def _scale_data(self, data_train, data_test, feature):
        concatenated_data = np.concatenate(data_train[feature])
        mean = concatenated_data.mean(axis=0)
        std = concatenated_data.std(axis=0)

        for i in range(len(data_train[feature])):
            data_train[feature][i] = (data_train[feature][i] - mean) / std
        for i in range(len(data_test[feature])):
            data_test[feature][i] = (data_test[feature][i] - mean) / std
        return data_train, data_test

    def _sort_by_reach_direction(self, data):
        sorted_by_label = np.argsort(np.array([reach_dir[0, 0] for reach_dir in data['labels']]))
        for feature in data.keys():
            data[feature] = np.array(data[feature])[sorted_by_label]
        return data

    def _map_labels(self, data):
        labels = data['labels']
        for i, l in enumerate(np.unique(labels)):
            labels[data['labels']==l] = i
        return labels


def get_class_data(dataset, device='cpu'):
    def get_data():
        firing_rates = dataset.firing_rates
        labels = dataset.labels
        data = [torch.tensor(firing_rates, dtype=torch.float32, device=device),
                      torch.tensor(labels, dtype=torch.long, device=device)]
        return data
    dataset.train()
    data_train = get_data()

    dataset.test()
    data_test = get_data()

    dataset.train()
    return data_train, data_test


def get_angular_data(dataset, velocity_threshold=-1., device='cpu'):
    def get_data():
        velocity_mask = np.linalg.norm(dataset.velocity, 2, axis=1) > velocity_threshold
        firing_rates = dataset.firing_rates[velocity_mask]
        labels = dataset.labels[velocity_mask]

        angles = (2 * np.pi / 8 * labels)[:, np.newaxis]
        cos_sin = np.concatenate([np.cos(angles), np.sin(angles)], axis=1)
        data = [torch.tensor(firing_rates, dtype=torch.float32, device=device),
                torch.tensor(angles, dtype=torch.float32, device=device),
                torch.tensor(cos_sin, dtype=torch.float32, device=device)]
        return data
    dataset.train()
    data_train = get_data()

    dataset.test()
    data_test = get_data()

    dataset.train()
    return data_train, data_test
