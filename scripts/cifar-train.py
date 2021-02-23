import os

from absl import app
from absl import flags
import torch
import tensorflow as tf
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Compose
from torchvision import transforms
from tqdm import tqdm

from self_supervised.trainer import MYOWTrainer
from self_supervised.nets import resnet_cifar
from self_supervised.utils import set_random_seeds


FLAGS = flags.FLAGS

flags.DEFINE_float('lr', 0.02, 'Base learning rate.')
flags.DEFINE_float('mm', 0.99, 'Base momentum.')
flags.DEFINE_integer('lr_warmup_epochs', 10, 'Warmup period for learning rate.')
flags.DEFINE_integer('batch_size', 256, 'Batch size.')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight decay.')
flags.DEFINE_enum('optimizer', 'lars', ['lars', 'sgd'], 'Optimizer.')
flags.DEFINE_string('logdir', 'myow-run', 'Tensorboard dir name.')
flags.DEFINE_string('ckptpath', 'myow-model', 'Checkkpoint folder dir name.')
flags.DEFINE_string('port', '12355', 'Master port.')


def train(rank, world_size, args):
    gpu = rank
    master_gpu = 0
    # load dataset

    set_random_seeds(random_seed=100)

    pre_transform = Compose([ToTensor()])

    dataset = CIFAR10("./data/cifar10", train=True, download=True, transform=pre_transform,
                      target_transform=torch.tensor)

    # Class of transformation for BYOL
    image_size = 32

    transform = transforms.Compose([transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                                    transforms.RandomGrayscale(p=0.2),
                                    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

    transform_m = transforms.Compose([transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                                      transforms.RandomHorizontalFlip(p=0.5),
                                      transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

    def prepare_views(inputs):
        x, labels = inputs
        outputs = {'view1': x, 'view2': x}
        return outputs

    batch_size = args.batch_size
    num_workers = 4

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False,
                            drop_last=True, sampler=sampler, pin_memory=True)

    # build ResNET network
    encoder = resnet_cifar()
    representation_size = 512
    projection_hidden_size = 4096
    projection_size = 256

    # build byol trainer
    projection_size_2 = 64
    projection_hidden_size_2 = 1024
    lr = args.lr
    momentum = args.mm

    train_epochs = 800
    byol_warmup_epochs = args.lr_warmup_epochs
    myow_warmup_epochs = 100
    myow_rampup_epochs = 110
    myow_max_weight = 1.0

    if not os.path.exists('ckpt/{}'.format(args.ckptpath)):
        os.makedirs('ckpt/{}'.format(args.ckptpath))

    trainer = MYOWTrainer(encoder=encoder, representation_size=representation_size, projection_size=projection_size,
                          projection_hidden_size=projection_hidden_size, projection_size_2=projection_size_2,
                          projection_hidden_size_2=projection_hidden_size_2, prepare_views=prepare_views,
                          train_dataloader=dataloader, view_pool_dataloader=dataloader, transform=transform,
                          transform_m=transform_m,
                          total_epochs=train_epochs, warmup_epochs=byol_warmup_epochs,
                          myow_warmup_epochs=myow_warmup_epochs, myow_rampup_epochs=myow_rampup_epochs,
                          myow_max_weight=myow_max_weight, batch_size=batch_size,
                          base_lr=lr, base_momentum=momentum, momentum=0.9, weight_decay=args.weight_decay,
                          exclude_bias_and_bn=True, optimizer_type=args.optimizer, symmetric_loss=True, view_miner_k=1,
                          world_size=world_size, rank=rank, gpu=gpu, master_gpu=master_gpu, distributed=True,
                          port=args.port, decay='cosine', m_decay='cosine',
                          log_step=10, log_dir='runs-cifar/{}'.format(args.logdir),
                          ckpt_path='ckpt/{}/ckpt-%d.pt'.format(args.ckptpath))

    save_frequency = 5
    for epoch in tqdm(range(train_epochs + 1), position=rank, desc='gpu:%d' % rank):
        dataloader.sampler.set_epoch(epoch)
        trainer.train_epoch()
        if epoch % save_frequency == 0:
            trainer.save_checkpoint(epoch)
            dist.barrier()


def main():
    world_size = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = FLAGS.port
    mp.spawn(train, nprocs=world_size, args=(world_size, FLAGS), join=True)


if __name__ == "__main__":
    print(f'PyTorch version: {torch.__version__}')
    app.run(main)
