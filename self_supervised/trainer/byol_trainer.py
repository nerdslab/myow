import os
import warnings

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD, Adam
from torch.nn.parallel import DistributedDataParallel as DDP

from self_supervised.model import BYOL, MLP3
from self_supervised.optimizer import LARS
from self_supervised.loss import CosineLoss


class BYOLTrainer():
    def __init__(self,
                 encoder, representation_size, projection_size, projection_hidden_size,
                 train_dataloader, prepare_views, total_epochs, warmup_epochs, base_lr, base_momentum,
                 batch_size=256, decay='cosine', n_decay=1.5, m_decay='cosine',
                 optimizer_type="lars", momentum=1.0, weight_decay=1.0, exclude_bias_and_bn=False,
                 transform=None, transform_1=None, transform_2=None, symmetric_loss=False,
                 world_size=1, rank=0, distributed=False, gpu=0, master_gpu=0, port='12355',
                 ckpt_path="./models/ckpt-%d.pt", log_step=1, log_dir=None, **kwargs):

        # device parameters
        self.world_size = world_size
        self.rank = rank
        self.gpu = gpu
        self.master_gpu = master_gpu
        self.distributed = distributed

        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{self.gpu}')
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device('cpu')

        print('Using %r.' %self.device)

        # checkpoint
        self.ckpt_path = ckpt_path

        # build network
        self.representation_size = representation_size
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size
        self.model = self.build_model(encoder)

        if self.distributed:
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = port
            dist.init_process_group(backend='nccl', init_method='env://', rank=self.rank, world_size=self.world_size)
            self.group = dist.new_group()

            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = DDP(self.model, device_ids=[self.gpu], find_unused_parameters=True)

        # dataloaders
        self.train_dataloader = train_dataloader
        self.prepare_views = prepare_views # outputs view1 and view2 (pre-gpu-transform)

        # transformers
        # these are on gpu transforms! can have cpu transform in dataloaders
        self.transform_1 = transform_1 if transform_1 is not None else transform # class 1 of transformations
        self.transform_2 = transform_2 if transform_2 is not None else transform # class 2 of transformations
        assert (self.transform_1 is None) == (self.transform_2 is None)

        # training parameters
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs

        # todo fix batch shape (double batch loader)
        self.train_batch_size = batch_size
        self.global_batch_size = self.world_size * self.train_batch_size

        self.num_examples = len(self.train_dataloader.dataset)
        self.warmup_steps = self.warmup_epochs * self.num_examples // self.global_batch_size
        self.total_steps = self.total_epochs * self.num_examples // self.global_batch_size

        self.step = 0
        base_lr = base_lr / 256
        self.max_lr = base_lr * self.global_batch_size

        self.base_mm = base_momentum

        assert decay in ['cosine', 'poly']
        self.decay = decay
        self.n_decay = n_decay

        assert m_decay in ['cosine', 'cste']
        self.m_decay = m_decay

        # configure optimizer
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.exclude_bias_and_bn = exclude_bias_and_bn

        if self.exclude_bias_and_bn:
            if not self.distributed:
                params = self._collect_params(self.model.trainable_modules)
            else:
                # todo make sure this is correct
                params = self._collect_params(self.model.module.trainable_module_list)
        else:
            params = self.model.parameters()

        if optimizer_type == "lars":
            self.optimizer = LARS(params, lr=self.max_lr, momentum=self.momentum, weight_decay=self.weight_decay)
        elif optimizer_type == "sgd":
            self.optimizer = SGD(params, lr=base_lr, momentum=self.momentum, weight_decay=self.weight_decay)
        elif optimizer_type == "adam":
            if momentum != 1.0:
                warnings.warn("Adam optimizer doesn't use momentum. Momentum %.2f will be ignored." % momentum)
            self.optimizer = Adam(params, lr=base_lr, weight_decay=self.weight_decay)
        else:
            raise ValueError("Optimizer type needs to be 'lars', 'sgd' or 'adam', got (%s)." % optimizer_type)

        self.loss = CosineLoss().to(self.device)
        self.symmetric_loss = symmetric_loss

        # logging
        self.log_step = log_step
        if self.rank == 0:
            self.writer = SummaryWriter(log_dir)

    def build_model(self, encoder):
        projector = MLP3(self.representation_size, self.projection_size, self.projection_hidden_size)
        predictor = MLP3(self.projection_size, self.projection_size, self.projection_hidden_size)
        net = BYOL(encoder, projector, predictor)
        return net.to(self.device)

    def _collect_params(self, model_list):
        """
        exclude_bias_and bn: exclude bias and bn from both weight decay and LARS adaptation
            in the PyTorch implementation of ResNet, `downsample.1` are bn layers
        """
        param_list = []
        for model in model_list:
            for name, param in model.named_parameters():
                if self.exclude_bias_and_bn and ('bn' in name or 'downsample.1' in name or 'bias' in name):
                    param_dict = {'params': param, 'weight_decay': 0., 'lars_exclude': True}
                else:
                    param_dict = {'params': param}
                param_list.append(param_dict)
        return param_list

    def _cosine_decay(self, step):
        return 0.5 * self.max_lr * (1 + np.cos((step - self.warmup_steps) * np.pi / (self.total_steps - self.warmup_steps)))

    def _poly_decay(self, step):
        return self.max_lr * (1 - ((step - self.warmup_steps) / (self.total_steps- self.warmup_steps)) ** self.n_decay)

    def update_learning_rate(self, step, decay='poly'):
        """learning rate warm up and decay"""
        if step <= self.warmup_steps:
            lr = self.max_lr * step / self.warmup_steps
        else:
            if self.decay == 'cosine':
                lr = self._cosine_decay(step)
            elif self.decay == 'poly':
                lr = self._poly_decay(step)
            else:
                raise AttributeError
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def update_momentum(self, step):
        if self.m_decay == 'cosine':
            self.mm = 1 - (1 - self.base_mm) * (np.cos(np.pi * step / self.total_steps) + 1) / 2
        elif self.m_decay == 'cste':
            self.mm = self.base_mm
        else:
            raise AttributeError

    def save_checkpoint(self, epoch):
        if self.rank == 0:
            state = {
                     'epoch': epoch,
                     'steps': self.step,
                     'model': self.model.state_dict(),
                     'optimizer': self.optimizer.state_dict(),
                    }
            torch.save(state, self.ckpt_path %(epoch))

    def load_checkpoint(self, epoch):
        model_path = self.ckpt_path %(epoch)
        map_location = {"cuda:{}": "cuda:{}".format(self.master_gpu, self.gpu)}
        map_location = "cuda:{}".format(self.gpu)
        checkpoint = torch.load(model_path, map_location=map_location)

        self.step = checkpoint['steps']
        self.model.load_state_dict(checkpoint['model'], strict=False)

        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def cleanup(self):
        dist.destroy_process_group()

    def forward_loss(self, preds, targets):
        loss = self.loss(preds, targets)
        return loss

    def update_target_network(self):
        if not self.distributed:
            self.model.update_target_network(self.mm)
        else:
            self.model.module.update_target_network(self.mm)

    def log_schedule(self, loss):
        self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], self.step)
        self.writer.add_scalar('mm', self.mm, self.step)
        self.writer.add_scalar('loss', loss, self.step)

    def train_epoch(self):
        self.model.train()
        for inputs in self.train_dataloader:
            # update parameters
            self.update_learning_rate(self.step)
            self.update_momentum(self.step)

            inputs = self.prepare_views(inputs)
            view1 = inputs['view1'].to(self.device)
            view2 = inputs['view2'].to(self.device)

            if self.transform_1 is not None:
                # apply transforms
                view1 = self.transform_1(view1)
                view2 = self.transform_2(view2)

            # forward
            outputs = self.model({'online_view': view1, 'target_view':view2})
            loss = self.forward_loss(outputs['online_q'], outputs['target_z'])
            if self.symmetric_loss:
                outputs = self.model({'online_view': view2, 'target_view': view1})
                loss += self.forward_loss(outputs['online_q'], outputs['target_z'])
                loss /= 2

            # backprop online network
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # update moving average
            self.update_target_network()

            # log
            if self.step % self.log_step == 0 and self.rank == 0:
                self.log_schedule(loss=loss.item())

            # update parameters
            self.step += 1
