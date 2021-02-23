from torch import nn


class MLP(nn.Module):
    r"""Multi-layer perceptron model, with optional regularization layers.

    Args:
        hidden_layers (list): List of layer dimensions, from input layer to output layer.
        activation (torch.nn.Module, optional): Activation function. (default: :obj:`nn.ReLU`).
        batchnorm (boolean, optional): If set to :obj:`True`, batchnorm layers are added after each block.
            (default: :obj:`False`).
        bias (boolean, optional): If set to :obj:`True`, bias will be used in linear layers. (default: :obj:`True`).
        drop_last_nonlin (boolean, optional): If set to :obj:`True`, the last layer won't have non-linearities or
            regularization layers. (default: :obj:`True`)
    """
    def __init__(self, hidden_layers, activation=nn.ReLU(True), batchnorm=False, bias=True, drop_last_nonlin=True):
        super().__init__()

        # build the layers
        layers = []
        for in_dim, out_dim in zip(hidden_layers[:-1], hidden_layers[1:]):
            layers.append(nn.Linear(in_dim, out_dim, bias=bias))
            if batchnorm:
                layers.append(nn.BatchNorm1d(num_features=out_dim))
            if activation is not None:
                layers.append(activation)

        # remove activation and/or batchnorm layers from the last block
        if drop_last_nonlin:
            remove_layers = -(int(activation is not None) + int(batchnorm))
            if remove_layers:
                layers = layers[:remove_layers]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x
