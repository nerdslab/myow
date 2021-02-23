import torch
import torch.nn.functional as F


def myow_factory(byol_class):
    r"""Factory function for adding mining feature to an architecture."""
    class MYOW(byol_class):
        r"""
        Class that adds ability to mine views to base class :obj:`byol_class`.

        Args:
            n_neighbors (int, optional): Number of neighbors used in knn. (default: :obj:`1`)
        """

        def __init__(self, *args, n_neighbors=1):
            super().__init__(*args)

            self.k = n_neighbors

        def _compute_distance(self, x, y):
            x = F.normalize(x, dim=-1, p=2)
            y = F.normalize(y, dim=-1, p=2)

            dist = 2 - 2 * torch.sum(x.view(x.shape[0], 1, x.shape[1]) *
                                     y.view(1, y.shape[0], y.shape[1]), -1)
            return dist

        def _knn(self, x, y):
            # compute distance
            dist = self._compute_distance(x, y)

            # compute k nearest neighbors
            values, indices = torch.topk(dist, k=self.k, largest=False)

            # randomly select one of the neighbors
            selection_mask = torch.randint(self.k, size=(indices.size(0),))
            mined_views_ids = indices[torch.arange(indices.size(0)).to(selection_mask), selection_mask]
            return mined_views_ids

        def mine_views(self, y, y_pool):
            r"""Finds, for each element in batch :obj:`y`, its nearest neighbors in :obj:`y_pool`, randomly selects one
                of them and returns the corresponding index.

            Args:
                y (torch.Tensor): batch of representation vectors.
                y_pool (torch.Tensor): pool of candidate representation vectors.

            Returns:
                torch.Tensor: Indices of mined views in :obj:`y_pool`.
            """
            mined_views_ids = self._knn(y, y_pool)
            return mined_views_ids
    return MYOW
