from torch import nn


class MLP3(nn.Module):
    r"""MLP class used for projector and predictor in :class:`BYOL`. The MLP has one hidden layer.

    .. note::

        The hidden layer should be larger than both input and output layers, according to the
        :class:`BYOL` paper.

    Args:
        input_size (int): Size of input features.
        output_size (int): Size of output features (projection or prediction).
        hidden_size (int): Size of hidden layer.
    """
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)
