from .mlp3 import MLP3
from .byol import BYOL
from .double_byol import DoubleBYOL
from .myow_factory import myow_factory

# generate myow variants
MYOW = myow_factory(DoubleBYOL)

__all__ = [
    'BYOL',
    'MLP3',
    'MYOW'
]
