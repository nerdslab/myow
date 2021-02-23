import numpy as np
import torch
import torch.distributed as dist

from self_supervised.model import MYOW, MLP3
from self_supervised.trainer import BYOLTrainer


class MYOWTrainer(BYOLTrainer):
    def __init__(self, view_pool_dataloader=None, transform_m=None,
                 myow_warmup_epochs=0, myow_rampup_epochs=None, myow_max_weight=1., view_miner_k=4,
                 log_img_step=0, untransform_vis=None, projection_size_2=None, projection_hidden_size_2=None, **kwargs):

        self.projection_size_2 = projection_size_2 if projection_size_2 is not None else kwargs['projection_size']
        self.projection_hidden_size_2 = projection_hidden_size_2 if projection_hidden_size_2 is not None \
            else kwargs['projection_hidden_size']

        # view pool dataloader
        self.view_pool_dataloader = view_pool_dataloader

        # view miner
        self.view_miner_k = view_miner_k

        # transform class for minning
        self.transform_m = transform_m

        # myow loss
        self.mined_loss_weight = 0.
        self.myow_max_weight = myow_max_weight
        self.myow_warmup_epochs = myow_warmup_epochs if myow_warmup_epochs is not None else 0
        self.myow_rampup_epochs = myow_rampup_epochs if myow_rampup_epochs is not None else kwargs['total_epochs']

        # convert to steps
        world_size = kwargs['world_size'] if 'world_size' in kwargs else 1
        self.num_examples = len(kwargs['train_dataloader'].dataset)
        self.train_batch_size = kwargs['batch_size']
        self.global_batch_size = world_size * self.train_batch_size
        self.myow_warmup_steps = self.myow_warmup_epochs * self.num_examples // self.global_batch_size
        self.myow_rampup_steps = self.myow_rampup_epochs * self.num_examples // self.global_batch_size
        self.total_steps = kwargs['total_epochs'] * self.num_examples // self.global_batch_size

        # logger
        self.log_img_step = log_img_step
        self.untransform_vis = untransform_vis

        super().__init__(**kwargs)

    def build_model(self, encoder):
        projector_1 = MLP3(self.representation_size, self.projection_size, self.projection_hidden_size)
        projector_2 = MLP3(self.projection_size, self.projection_size_2, self.projection_hidden_size_2)
        predictor_1 = MLP3(self.projection_size, self.projection_size, self.projection_hidden_size)
        predictor_2 = MLP3(self.projection_size_2, self.projection_size_2, self.projection_hidden_size_2)
        net = MYOW(encoder, projector_1, projector_2, predictor_1, predictor_2, n_neighbors=self.view_miner_k)
        return net.to(self.device)

    def update_mined_loss_weight(self, step):
        max_w = self.myow_max_weight
        min_w = 0.
        if step < self.myow_warmup_steps:
            self.mined_loss_weight = min_w
        elif step > self.myow_rampup_steps:
            self.mined_loss_weight = max_w
        else:
            self.mined_loss_weight = min_w + (max_w - min_w) * (step - self.myow_warmup_steps) / \
                                     (self.myow_rampup_steps - self.myow_warmup_steps)

    def log_schedule(self, loss):
        super().log_schedule(loss)
        self.writer.add_scalar('myow_weight', self.mined_loss_weight, self.step)

    def log_correspondance(self, view, view_mined):
        """ currently only implements 2d images"""
        img_batch = np.zeros((16, view.shape[1], view.shape[2], view.shape[3]))
        for i in range(8):
            img_batch[i] = self.untransform_vis(view[i]).detach().cpu().numpy()
            img_batch[8+i] = self.untransform_vis(view_mined[i]).detach().cpu().numpy()
        self.writer.add_images('correspondence', img_batch, self.step)

    def train_epoch(self):
        self.model.train()
        if self.view_pool_dataloader is not None:
            view_pooler = iter(self.view_pool_dataloader)
        for inputs in self.train_dataloader:
            # update parameters
            self.update_learning_rate(self.step)
            self.update_momentum(self.step)
            self.update_mined_loss_weight(self.step)
            self.optimizer.zero_grad()

            inputs = self.prepare_views(inputs)
            view1 = inputs['view1'].to(self.device)
            view2 = inputs['view2'].to(self.device)

            if self.transform_1 is not None:
                # apply transforms
                view1 = self.transform_1(view1)
                view2 = self.transform_2(view2)

            # forward
            outputs = self.model({'online_view': view1, 'target_view':view2})
            weight = 1 / (1. + self.mined_loss_weight)
            if self.symmetric_loss:
                weight /= 2.
            loss = weight * self.forward_loss(outputs['online_q'], outputs['target_z'])

            if self.distributed and self.mined_loss_weight > 0 and not self.symmetric_loss:
                with self.model.no_sync():
                    loss.backward()
            else:
                loss.backward()

            if self.symmetric_loss:
                outputs = self.model({'online_view': view2, 'target_view': view1})
                weight = 1 / (1. + self.mined_loss_weight) / 2.
                loss = weight * self.forward_loss(outputs['online_q'], outputs['target_z'])
                if self.distributed and self.mined_loss_weight > 0:
                    with self.model.no_sync():
                        loss.backward()
                else:
                    loss.backward()

            # mine view
            if self.mined_loss_weight > 0:
                if self.view_pool_dataloader is not None:
                    try:
                        # currently only supports img, label
                        view_pool, label_pool = next(view_pooler)
                        view_pool = view_pool.to(self.device).squeeze()
                    except StopIteration:
                        # reinit the dataloader
                        view_pooler = iter(self.view_pool_dataloader)
                        view_pool, label_pool = next(view_pooler)
                        view_pool = view_pool.to(self.device).squeeze()
                    view3 = inputs['view1'].to(self.device)
                else:
                    view3 = inputs['view3'].to(self.device).squeeze() \
                        if 'view3' in inputs else inputs['view1'].to(self.device).squeeze()
                    view_pool = inputs['view_pool'].to(self.device).squeeze()

                # apply transform
                if self.transform_m is not None:
                    # apply transforms
                    view3 = self.transform_m(view3)
                    view_pool = self.transform_m(view_pool)

                # compute representations
                outputs = self.model({'online_view': view3}, get_embedding='encoder')
                online_y = outputs['online_y']
                outputs_pool = self.model({'target_view': view_pool}, get_embedding='encoder')
                target_y_pool = outputs_pool['target_y']

                # mine views
                if self.distributed:
                    gather_list = [torch.zeros_like(target_y_pool) for _ in range(self.world_size)]
                    dist.all_gather(gather_list, target_y_pool, self.group)
                    target_y_pool = torch.cat(gather_list, dim=0)
                    selection_mask = self.model.module.mine_views(online_y, target_y_pool)
                else:
                    selection_mask = self.model.mine_views(online_y, target_y_pool)

                target_y_mined = target_y_pool[selection_mask].contiguous()
                outputs_mined = self.model({'online_y': online_y,'target_y': target_y_mined}, get_embedding='predictor_m')
                weight = self.mined_loss_weight / (1. + self.mined_loss_weight)
                loss = weight * self.forward_loss(outputs_mined['online_q_m'], outputs_mined['target_v'])
                loss.backward()

            self.optimizer.step()

            # update moving average
            self.update_target_network()

            # log
            if self.step % self.log_step == 0 and self.rank == 0:
                self.log_schedule(loss=loss.item())

            # log images
            if self.mined_loss_weight > 0 and self.log_img_step > 0 and self.step % self.log_img_step == 0 and self.rank == 0:
                if self.distributed:
                    # get image pools from all gpus
                    gather_list = [torch.zeros_like(view_pool) for _ in range(self.world_size)]
                    dist.all_gather(gather_list, view_pool, self.group)
                    view_pool = torch.cat(gather_list, dim=0)
                self.log_correspondance(view3, view_pool[selection_mask])

            # update parameters
            self.step += 1

        return loss.item()
