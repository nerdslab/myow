import torch
from tqdm import tqdm

from self_supervised.data import utils
from self_supervised.utils import MetricLogger


def compute_representations(net, dataloader, device='cpu'):
    r"""Pre-computes the representation for the entire dataset.

    Args:
        net (torch.nn.Module): Frozen encoder.
        dataloader (torch.data.DataLoader): Dataloader.
        device (String, Optional): Device used. (default: :obj:`"cpu"`)

    Returns:
        [torch.Tensor, torch.Tensor]: Representations and labels.
    """
    net.eval()
    reps = []
    labels = []

    for i, (x, label) in tqdm(enumerate(dataloader)):
        # load data
        x = x.to(device).squeeze()
        labels.append(label)

        # forward
        with torch.no_grad():
            representation = net(x)
            reps.append(representation.detach().cpu().squeeze())

        if i % 10 == 0:
            reps = [torch.cat(reps, dim=0)]
            labels = [torch.cat(labels, dim=0)]
            
    reps = torch.cat(reps, dim=0)
    labels = torch.cat(labels, dim=0)
    return [reps, labels]


def compute_accuracy(classifier, data, batch_size=256, device='cpu'):
    r"""Evaluates the classification accuracy with representations pre-computed.

    Args:
        classifier (torch.nn.Module): Linear layer.
        data (list of torch.nn.Tensor): Inputs, target class and target angles.
        batch_size (int, Optional): Batch size used during evaluation. It has no impact on final accuracy.
            (default: :obj:`256`)
        device (String, Optional): Device used. (default: :obj:`"cpu"`)

    Returns:
        float: Accuracy.
    """
    # prepare inputs
    classifier.eval()
    right = []
    total = []
    for x, label in utils.batch_iter(*data, batch_size=batch_size):
        x = x.to(device)
        label = label.to(device)

        # feed to network and classifier
        with torch.no_grad():
            pred_logits = classifier(x)
        # compute accuracy
        _, pred_class = torch.max(pred_logits, 1)
        right.append((pred_class == label).sum().item())
        total.append(label.size(0))
    classifier.train()
    return sum(right) / sum(total)


def train_linear_layer(classifier, data_train, data_val, optimizer, scheduler=None,
                       batch_size=256, num_epochs=10, device='cpu', writer=None, tag="", tqdm_progress=False):
    r"""Trains linear layer to predict angle with representation pre-computed.

    Args:
        classifier (torch.nn.Module): Trainable linear layer.
        data_train (torch.data.DataLoader or list of torch.nn.Tensor): Representations and target class.
        data_val (torch.data.DataLoader or list of torch.nn.Tensor): Representations and target class.
        optimizer (torch.optim.Optimizer): Optimizer for :obj:`classifier`.
        scheduler (torch.optim._LRScheduler, Optional): Learning rate scheduler. (default: :obj:`None`)
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

    acc = MetricLogger(smoothing_factor=0.2)
    for epoch in tqdm(range(num_epochs), disable=not tqdm_progress):
        for x, label in utils.batch_iter(*data_train, batch_size=batch_size):
            classifier.train()
            optimizer.zero_grad()

            # load data
            x = x.to(device)
            label = label.to(device)

            pred_class = classifier(x)

            # loss
            loss = class_criterion(pred_class, label)

            # backward
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # compute classification accuracies
        acc_val = compute_accuracy(classifier, data_val, batch_size=batch_size, device=device)
        acc.update(0., acc_val)
        if writer is not None:
            writer.add_scalar('eval_acc/val-%r' %tag, acc_val, epoch)

    acc_train = compute_accuracy(classifier, data_train, batch_size=batch_size, device=device)
    acc.update(acc_train, acc_val)
    return acc
