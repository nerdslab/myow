import copy

import torch

from .byol import BYOL


class DoubleBYOL(BYOL):
    r"""BYOL with dual projector/predictor networks.

    When the projectors are cascaded, the two views are separately forwarded through the online and target networks:
    .. math::
        y = f_{\theta}(x),\  z = g_{\theta}(y), v = h_{\theta}(z)\\
        y^\prime = f_{\xi}(x^\prime),\  z^\prime = g_{\xi}(y^\prime), v^\prime = h_{\xi}(z^\prime)

    then prediction is performed either in the first projection space or the second.
    In the first, the predictor learns to predict the target projection from the online projection in order to minimize
        the following loss:
    .. math::
        \mathcal{L}_{\theta, \xi} = 2-2 \cdot \frac{\left\langle q_{\theta}\left(z\right),
        z^{\prime}\right\rangle}{\left\|q_{\theta}\left(z\right)\right\|_{2}
        \cdot\left\|z^{\prime}\right\|_{2}}

    In the second, the second predictor learns to predict the second target projection from the second online projection
        in order to minimize the following loss:
    .. math::
        \mathcal{L}_{\theta, \xi} = 2-2 \cdot \frac{\left\langle r_{\theta}\left(v\right),
        v^{\prime}\right\rangle}{\left\|v_{\theta}\left(v\right)\right\|_{2}
        \cdot\left\|v^{\prime}\right\|_{2}}.

    Args:
        encoder (torch.nn.Module): Encoder network to be duplicated and used in both online and target networks.
        projector (torch.nn.Module): Projector network to be duplicated and used in both online and target networks.
        projector_m (torch.nn.Module): Second projector network to be duplicated and used in both online and
            target networks.
        predictor (torch.nn.Module): Predictor network used to predict the target projection from the online projection.
        predictor_m (torch.nn.Module): Second predictor network used to predict the target projection from the
            online projection.
        layout (String, Optional): Defines the layout of the dual projectors. Can be either :obj:`"cascaded"` or
            :obj:`"parallel"`. (default: :obj:`"cascaded"`)
    """
    def __init__(self, encoder, projector, projector_m, predictor, predictor_m, layout='cascaded'):
        super().__init__(encoder, projector, predictor)

        assert layout in ['cascaded', 'parallel'], "layout should be 'cascaded' or 'parallel', got %s." % layout
        self.layout = layout

        self.online_projector_m = projector_m
        self.target_projector_m = copy.deepcopy(projector_m)
        self._stop_gradient(self.target_projector_m)

        self.predictor_m = predictor_m

    @property
    def trainable_modules(self):
        r"""Returns the list of modules that will updated via an optimizer."""
        return [self.online_encoder, self.online_projector, self.online_projector_m, self.predictor, self.predictor_m]

    @property
    def _ema_module_pairs(self):
        return [(self.online_encoder, self.target_encoder),
                (self.online_projector, self.target_projector),
                (self.online_projector_m, self.target_projector_m)]

    def forward(self, inputs, get_embedding='predictor'):
        r"""Defines the computation performed at every call. Supports single or dual forwarding through online and/or
        target networks. Supports resuming computation from encoder space.

        :obj:`get_embedding` determines whether the computation stops at the encoder space (:obj:`"encoder"`) or at
            the first projector space (:obj:`"predictor"`) or the second (:obj:`"predictor_m"`). With the last two
            options, predictions are also made.

        :obj:`inputs` can include :obj:`"online_view"` and/or :obj:`"target_view"`. The prefix determines which
        branch the tensor is passed through. It is possible to give one view only, which will be forwarded through
        its corresponding branch.

        To resume computation from the encoder space, simply pass :obj:`"online_y"` and/or :obj:`"target_y"`. If, for
        example, :obj:`"online_y"` is present in :obj:`inputs` then :obj:`"online_view"`, if passed, would be ignored.

        Args:
            inputs (dict): Inputs to be forwarded through the networks.
            get_embedding (String, Optional): Determines where the computation stops, can be :obj:`"encoder"`,
                :obj:`"predictor"` or :obj:`"predictor_m"`. (default: :obj:`"predictor"`)

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

            inputs = {'online_view': x1, 'target_view': x2}
            outputs = net(inputs, get_embedding='predictor_m') # outputs online_q_m and target_v
        """
        assert get_embedding in ['encoder', 'predictor', 'predictor_m'], \
            "Module name needs to be in %r." % ['encoder', 'predictor', 'predictor_m']

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

            if get_embedding == 'predictor_m':
                if self.layout == 'parallel':
                    online_v = self.online_projector_m(online_y)
                    online_q_m = self.predictor_m(online_v)

                    outputs['online_q_m'] = online_q_m

                elif self.layout == 'cascaded':
                    online_z = self.online_projector(online_y)
                    online_v = self.online_projector_m(online_z)
                    online_q_m = self.predictor_m(online_v)

                    outputs['online_q_m'] = online_q_m

        if 'target_view' in inputs or 'target_y' in inputs:
            # forward target encoder
            with torch.no_grad():
                if not ('target_y' in inputs):
                    # representation is not already computed, requires forwarding the view through the target encoder.
                    target_view = inputs['target_view']
                    target_y = self.target_encoder(target_view)
                    target_y = target_y.view(target_y.shape[0], -1).contiguous()
                else:
                    # resume forwarding
                    target_y = inputs['target_y']

                if get_embedding == 'encoder':
                    outputs['target_y'] = target_y

                if get_embedding == 'predictor':
                    # forward projector and predictor
                    target_z = self.target_projector(target_y).detach().clone()

                    outputs['target_z'] = target_z

                if get_embedding == 'predictor_m':
                    if self.layout == 'parallel':
                        target_v = self.target_projector_m(target_y)

                        outputs['target_v'] = target_v

                    elif self.layout == 'cascaded':
                        target_z = self.target_projector(target_y)
                        target_v = self.target_projector_m(target_z)

                        outputs['target_v'] = target_v
        return outputs
