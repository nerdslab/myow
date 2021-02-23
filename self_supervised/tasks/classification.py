import torch
from tqdm import tqdm

from self_supervised.data import utils
from self_supervised.utils import MetricLogger


def compute_accuracy(net, classifier, data, transform=None, device='cpu'):
    r"""Evaluates the classification accuracy when a list of :class:`torch.Tensor` is given.

    Args:
        net (torch.nn.Module): Frozen encoder.
        classifier (torch.nn.Module): Linear layer.
        data (list of torch.nn.Tensor): Inputs, target class and target angles.
        transform (Callable, Optional): Transformation to use. Added for the purposes of
            normalization. (default: :obj:`None`)
        device (String, Optional): Device used. (default: :obj:`"cpu"`)

    Returns:
        float: Accuracy.
    """
    classifier.eval()
    # prepare inputs
    x, label = data
    x = x.to(device)
    label = label.to(device)

    if transform is not None:
        x = transform(x)

    # feed to network and classifier
    with torch.no_grad():
        representation = net(x)
        pred_logits = classifier(representation)

    # compute accuracy
    _, pred_class = torch.max(pred_logits, 1)
    acc = (pred_class == label).sum().item() / label.size(0)
    return acc


def compute_accuracy_dataloader(net, classifier, dataloader, transform=None, device='cpu'):
    r"""Evaluates the classification accuracy when a :obj:`torch.data.DataLoader` is given.

    Args:
        net (torch.nn.Module): Frozen encoder.
        classifier (torch.nn.Module): Linear layer.
        dataloader (torch.data.DataLoader): Dataloader.
        transform (Callable, Optional): Transformation to use. Added for the purposes of
            normalization. (default: :obj:`None`)
        device (String, Optional): Device used. (default: :obj:`"cpu"`)

    Returns:
        float: Accuracy.
    """
    classifier.eval()
    acc = []
    for x, label in dataloader:
        x = x.to(device)
        label = label.to(device)

        if transform is not None:
            x = transform(x)

        # feed to network and classifier
        with torch.no_grad():
            representation = net(x)
            representation = representation.view(representation.shape[0], -1)
            pred_logits = classifier(representation)
        # compute accuracy
        _, pred_class = torch.max(pred_logits, 1)
        acc.append((pred_class == label).sum().item() / label.size(0))
    return sum(acc)/len(acc)


def train_classifier(net, classifier, data_train, data_val, optimizer, scheduler=None, transform=None,
                     transform_val=None, batch_size=256, num_epochs=10, device='cpu',
                     writer=None, tag='', tqdm_progress=False):
    r"""Trains linear layer to predict angle.

    Args:
        net (torch.nn.Module): Frozen encoder.
        classifier (torch.nn.Module): Trainable linear layer.
        data_train (torch.data.DataLoader or list of torch.nn.Tensor): Inputs and target class.
        data_val (torch.data.DataLoader or list of torch.nn.Tensor): Inputs and target class.
        optimizer (torch.optim.Optimizer): Optimizer for :obj:`classifier`.
        scheduler (torch.optim._LRScheduler, Optional): Learning rate scheduler. (default: :obj:`None`)
        transform (Callable, Optional): Transformation to use during training. (default: :obj:`None`)
        transform_val (Callable, Optional): Transformation to use during validation. Added for the purposes of
            normalization. (default: :obj:`None`)
        batch_size (int, Optional): Batch size used during training. (default: :obj:`256`)
        num_epochs (int, Optional): Number of training epochs. (default: :obj:`10`)
        device (String, Optional): Device used. (default: :obj:`"cpu"`)
        writer (torch.utils.tensorboard.SummaryWriter, Optional): Summary writer. (default: :obj:`None`)
        tag (String, Optional): Tag used in :obj:`writer`. (default: :obj:`""`)
        tqdm_progress (bool, Optional): If :obj:`True`, show training progress.

    Returns:
        MetricLogger: Accuracy.
    """
    class_criterion = torch.nn.CrossEntropyLoss()

    acc = MetricLogger()
    for epoch in tqdm(range(num_epochs), disable=not tqdm_progress):
        classifier.train()
        if isinstance(data_train, list):
            iterator = utils.batch_iter(*data_train, batch_size=batch_size)
        else:
            iterator = iter(data_train)

        for x, label in iterator:
            optimizer.zero_grad()

            # load data
            x = x.to(device)
            label = label.to(device)

            if transform is not None:
                x = transform(x)

            # forward
            with torch.no_grad():
                representation = net(x)
                representation = representation.view(representation.shape[0], -1)

            pred_class = classifier(representation)

            # loss
            loss = class_criterion(pred_class, label)

            # backward
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # compute classification accuracies
        if isinstance(data_train, list):
            acc_val = compute_accuracy(net, classifier, data_val, transform=transform_val, device=device)
        else:
            acc_val = compute_accuracy_dataloader(net, classifier, data_val, transform=transform_val, device=device)

        acc.update(0., acc_val)
        if writer is not None:
            writer.add_scalar('eval_acc/val-%r' % tag, acc_val, epoch)

    if isinstance(data_train, list):
        acc_train = compute_accuracy(net, classifier, data_train, transform=transform_val, device=device)
    else:
        acc_train = compute_accuracy_dataloader(net, classifier, data_train, transform=transform_val, device=device)
    acc.update(acc_train, acc_val)
    return acc

