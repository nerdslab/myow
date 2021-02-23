import torch
from tqdm import tqdm
import numpy as np

from self_supervised.data import utils
from self_supervised.utils import MetricLogger


def compute_angle_accuracy(net, classifier, data, transform=None, device='cpu'):
    r"""Evaluates the angle prediction performance of the network.

    Args:
        net (torch.nn.Module): Frozen encoder.
        classifier (torch.nn.Module): Linear layer.
        data (list of torch.nn.Tensor): Inputs, target class and target angles.
        transform (Callable, Optional): Transformation to use. Added for the purposes of
            normalization. (default: :obj:`None`)
        device (String, Optional): Device used. (default: :obj:`"cpu"`)

    Returns:
        (float, float): Accuracy and delta-Accuracy.
    """
    # prepare inputs
    classifier.eval()
    x, a, y = data

    x = x.to(device).squeeze()
    a = a.to(device).squeeze()

    # transform data
    if transform is not None:
        [x] = transform(x)

    # feed to classifier
    with torch.no_grad():
        representation = net(x).detach()
        pred_cos_sin = classifier(representation).detach().clone()

    pred_angles = torch.atan2(pred_cos_sin[:, 1], pred_cos_sin[:, 0])
    pred_angles[pred_angles < 0] = pred_angles[pred_angles < 0] + 2 * np.pi

    diff_angles = torch.abs(pred_angles - a.squeeze())
    diff_angles[diff_angles > np.pi] = torch.abs(diff_angles[diff_angles > np.pi] - 2 * np.pi)

    error = 0.
    acc = (diff_angles < (np.pi / 8)).sum()
    acc = acc.item() / x.size(0)
    delta_acc = (diff_angles < (3 * np.pi / 16)).sum()
    delta_acc = delta_acc.item() / x.size(0)
    return acc, delta_acc


def train_angle_classifier(net, classifier, data_train, data_val, optimizer, transform=None,
                           transform_val=None, batch_size=256, num_epochs=10, device='cpu'):
    r"""Trains linear layer to predict angle.

    Args:
        net (torch.nn.Module): Frozen encoder.
        classifier (torch.nn.Module): Trainable linear layer.
        data_train (list of torch.nn.Tensor): Inputs, target class and target angles.
        data_val (list of torch.nn.Tensor): Inputs, target class and target angles.
        optimizer (torch.optim.Optimizer): Optimizer for :obj:`classifier`.
        transform (Callable, Optional): Transformation to use during training. (default: :obj:`None`)
        transform_val (Callable, Optional): Transformation to use during validation. Added for the purposes of
            normalization. (default: :obj:`None`)
        batch_size (int, Optional): Batch size used during training. (default: :obj:`256`)
        num_epochs (int, Optional): Number of training epochs. (default: :obj:`10`)
        device (String, Optional): Device used. (default: :obj:`"cpu"`)

    Returns:
        (MetricLogger, MetricLogger): Accuracy and delta-Accuracy.
    """
    class_criterion = torch.nn.MSELoss()

    acc = MetricLogger()
    delta_acc = MetricLogger()

    for epoch in tqdm(range(num_epochs), disable=True):
        classifier.train()
        for x, _, label in utils.batch_iter(*data_train, batch_size=batch_size):
            x = x.to(device).squeeze()
            label = label.to(device).squeeze()

            # transform data
            if transform is not None:
                [x] = transform(x)

            optimizer.zero_grad()
            # forward
            with torch.no_grad():
                representation = net(x).detach().clone()
                representation = representation.view(representation.shape[0], -1)

            pred_class = classifier(representation)

            # loss
            loss = class_criterion(pred_class, label)

            # backward
            loss.backward()
            optimizer.step()

        # compute classification accuracies
        acc_train, delta_acc_train = compute_angle_accuracy(net, classifier, data_train, transform=transform_val,
                                                            device=device)
        acc_test, delta_acc_test = compute_angle_accuracy(net, classifier, data_val, transform=transform_val,
                                                          device=device)

        acc.update(acc_train, acc_test, step=epoch)
        delta_acc.update(delta_acc_train, delta_acc_test, step=epoch)
    return acc, delta_acc
