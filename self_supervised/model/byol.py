import copy

import torch


class BYOL(torch.nn.Module):
    r"""Base backbone-agnostic BYOL architecture.
    The BYOL architecture was proposed by Grill et al. in
    Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning
        https://arxiv.org/abs/2006.07733

    Two views are separately forwarded through the online and target networks:
    .. math::
        y = f_{\theta}(x),\  z = g_{\theta}(y)\\
        y^\prime = f_{\xi}(x^\prime),\  z^\prime = g_{\xi}(y^\prime)

    then the predictor learns to predict the target projection from the online projection in order to minimize the
        following loss:
    .. math::
        \mathcal{L}_{\theta, \xi} = 2-2 \cdot \frac{\left\langle q_{\theta}\left(z\right),
        z^{\prime}\right\rangle}{\left\|q_{\theta}\left(z\right)\right\|_{2}
        \cdot\left\|z^{\prime}\right\|_{2}}.

    Args:
        encoder (torch.nn.Module): Encoder network to be duplicated and used in both online and target networks.
        projector (torch.nn.Module): Projector network to be duplicated and used in both online and target networks.
        predictor (torch.nn.Module): Predictor network used to predict the target projection from the online projection.
    """
    def __init__(self, encoder, projector, predictor):
        super().__init__()
        self.online_encoder = encoder
        self.target_encoder = copy.deepcopy(encoder)
        self._stop_gradient(self.target_encoder)

        self.online_projector = projector
        self.target_projector = copy.deepcopy(projector)
        self._stop_gradient(self.target_projector)

        self.predictor = predictor

    @property
    def trainable_modules(self):
        r"""Returns the list of modules that will updated via an optimizer."""
        return [self.online_encoder, self.online_projector, self.predictor]

    @property
    def _ema_module_pairs(self):
        return [(self.online_encoder, self.target_encoder),
                (self.online_projector, self.target_projector)]

    def _stop_gradient(self, network):
        r"""Stops parameters of :obj:`network` of being updated through back-propagation."""
        for param in network.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def _reset_moving_average(self):
        r"""Resets target network to have the same parameters as the online network."""
        for online_module, target_module in self._ema_module_pairs:
            for param_q, param_k in zip(online_module.parameters(), target_module.parameters()):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False     # stop gradient

    @torch.no_grad()
    def update_target_network(self, mm):
        r"""Performs a momentum update of the target network's weights.

        Args:
            mm (float): Momentum used in moving average update.
        """
        assert 0.0 <= mm <= 1.0, "Momentum needs to be between 0.0 and 1.0, got %.5f" % mm
        for online_module, target_module in self._ema_module_pairs:
            for param_q, param_k in zip(online_module.parameters(), target_module.parameters()):
                param_k.data.mul_(mm).add_(param_q.data, alpha=1. - mm)

    def forward(self, inputs, get_embedding='predictor'):
        r"""Defines the computation performed at every call. Supports single or dual forwarding through online and/or
        target networks. Supports resuming computation from encoder space.

        :obj:`get_embedding` determines whether the computation stops at the encoder space (:obj:`"encoder"`) or all
        computations including the projections and the prediction (:obj:`"predictor"`).

        :obj:`inputs` can include :obj:`"online_view"` and/or :obj:`"target_view"`. The prefix determines which
        branch the tensor is passed through. It is possible to give one view only, which will be forwarded through
        its corresponding branch.

        To resume computation from the encoder space, simply pass :obj:`"online_y"` and/or :obj:`"target_y"`. If, for
        example, :obj:`"online_y"` is present in :obj:`inputs` then :obj:`"online_view"`, if passed, would be ignored.

        Args:
            inputs (dict): Inputs to be forwarded through the networks.
            get_embedding (String, Optional): Determines where the computation stops, can be :obj:`"encoder"` or
                :obj:`"predictor"`. (default: :obj:`"predictor"`)

        Returns:
            dict

        Example::
            net = BYOL(...)
            inputs = {'online_view': x1, 'target_view': x2}
            outputs = net(inputs) # outputs online_q and target_z

            inputs = {'online_view': x1}
            outputs = net(inputs, get_embedding='encoder') # outputs online_y

            inputs = {'online_y': y1, 'target_y': y2}
            outputs = net(inputs) # outputs online_q and target_z
        """
        assert get_embedding in ['encoder', 'predictor'], "Module name needs to be in %r." % ['encoder', 'predictor']

        outputs = {}
        if 'online_view' in inputs or 'online_y' in inputs:
            # forward online network
            if not('online_y' in inputs):
                # representation is not already computed, requires forwarding the view through the online encoder.
                online_view = inputs['online_view']
                online_y = self.online_encoder(online_view)
                online_y = online_y.view(online_y.shape[0], -1).contiguous()  # flatten
            else:
                # resume forwarding
                online_y = inputs['online_y']

            if get_embedding == 'encoder':
                outputs['online_y'] = online_y

            if get_embedding == 'predictor':
                online_z = self.online_projector(online_y)
                online_q = self.predictor(online_z)

                outputs['online_q'] = online_q

        if 'target_view' in inputs or 'target_y' in inputs:
            # forward target network
            with torch.no_grad():
                if not ('target_y' in inputs):
                    # representation is not already computed, requires forwarding the view through the target encoder.
                    target_view = inputs['target_view']
                    target_y = self.target_encoder(target_view)
                    target_y = target_y.view(target_y.shape[0], -1).contiguous() # flatten
                else:
                    # resume forwarding
                    target_y = inputs['target_y']

                if get_embedding == 'encoder':
                    outputs['target_y'] = target_y

                if get_embedding == 'predictor':
                    # forward projector and predictor
                    target_z = self.target_projector(target_y).detach().clone()

                    outputs['target_z'] = target_z
        return outputs
