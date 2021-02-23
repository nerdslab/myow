import os
import re
import time

from absl import app
from absl import flags
from glob import glob
import torch
import tensorflow as tf
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import ToTensor
from torchvision import transforms

from self_supervised.trainer import MYOWTrainer
from self_supervised.nets import resnet_cifar
from self_supervised.tasks import fast_classification


FLAGS = flags.FLAGS

flags.DEFINE_float('lr', 0.02, 'Base learning rate.')
flags.DEFINE_integer('resume_eval', 0, 'Epoch at which evaluation starts.')
flags.DEFINE_string('logdir', 'myow-run', 'Tensorboard dir name.')
flags.DEFINE_string('ckptpath', 'myow-model', 'Checkkpoint folder dir name.')


def eval(gpu, ckpt_epoch, args, dataset_class=CIFAR10, num_classes=10):
    # load dataset
    image_size = 32
    transform = transforms.Compose([transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

    pre_transform = transforms.Compose([ToTensor(),
                                        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

    dataset = dataset_class("../datasets/cifar10", train=True, download=True, transform=pre_transform,
                    target_transform=torch.tensor)

    dataset_val = dataset_class("../datasets/cifar10", train=False, download=True, transform=pre_transform,
                    target_transform=torch.tensor)

    # Class of transformation for BYOL
    batch_size = 1024
    num_workers = 2

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False,
                            drop_last=False, pin_memory=True)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, num_workers=num_workers, shuffle=False,
                                drop_last=False, pin_memory=True)

    def prepare_views(inputs):
        x, labels = inputs
        outputs = {'view1': x, 'view2': x}
        return outputs

    # build ResNET network
    encoder = resnet_cifar()
    representation_size = 512
    projection_hidden_size = 4096
    projection_size = 256

    # build byol trainer
    projection_size_2 = 64
    projection_hidden_size_2 = 1024
    lr = 0.08
    momentum = 0.99

    train_epochs = 800
    byol_warmup_epochs = 10
    n_decay = 1.5
    myow_warmup_epochs = 100
    myow_rampup_epochs = 110
    myow_max_weight = 1.0

    trainer = MYOWTrainer(encoder=encoder, representation_size=representation_size, projection_size=projection_size,
                          projection_hidden_size=projection_hidden_size, projection_size_2=projection_size_2,
                          projection_hidden_size_2=projection_hidden_size_2, prepare_views=prepare_views,
                          train_dataloader=dataloader, view_pool_dataloader=dataloader, transform=transform,
                          total_epochs=train_epochs, warmup_epochs=byol_warmup_epochs,
                          myow_warmup_epochs=myow_warmup_epochs, myow_rampup_epochs=myow_rampup_epochs,
                          myow_max_weight=myow_max_weight, batch_size=batch_size,
                          base_lr=lr, base_momentum=momentum, momentum=0.9, weight_decay=args.wd,
                          exclude_bias_and_bn=True, optimizer_type='sgd', symmetric_loss=True, view_miner_k=1,
                          gpu=gpu, distributed=True, decay='cosine', m_decay='cosine',
                          log_step=10, log_dir='runs-cifar/{}-elr{}'.format(args.logdir, args.lr),
                          ckpt_path='ckpt/{}/ckpt-%d.pt'.format(args.ckptpath))

    trainer.load_checkpoint(ckpt_epoch)
    print('checkpoint loaded')

    trainer.model.eval()
    # representation
    print('computing representations')
    data_train = fast_classification.compute_representations(trainer.model.online_encoder.eval(), dataloader,
                                                             device=trainer.device)
    data_val = fast_classification.compute_representations(trainer.model.online_encoder.eval(), dataloader_val,
                                                           device=trainer.device)

    clr = args.lr
    classifier = resnet_cifar.get_linear_classifier(output_dim=num_classes).to(trainer.device)
    class_optimizer = torch.optim.SGD(classifier.parameters(), lr=clr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(class_optimizer, milestones=[60, 80], gamma=0.1)
    batch_size = 512
    acc = fast_classification.train_linear_layer(classifier, data_train, data_val, class_optimizer, scheduler=scheduler,
                                                 writer=trainer.writer, tag=ckpt_epoch, batch_size=batch_size,
                                                 num_epochs=100, device=trainer.device, tqdm_progress=True)

    print('Train', acc.train_last, ', Test', acc.val_smooth)
    trainer.writer.add_scalar('eval-train-%d' % num_classes, acc.train_last, ckpt_epoch)
    trainer.writer.add_scalar('eval-test-%d' % num_classes, acc.val_smooth, ckpt_epoch)


def find_checkpoints(ckpt_path):
    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        """alist.sort(key=natural_keys) sorts in human order
        http://nedbatchelder.com/blog/200712/human_sorting.html
        """
        return [atoi(c) for c in re.split(r'(\d+)', text)]

    ckpt_list = glob(os.path.join(ckpt_path, 'byol-ckpt-*.pt'))
    ckpt_list.sort(key=natural_keys)
    return ckpt_list

def main():
    ckptpath = 'ckpt/{}'.format(FLAGS.ckptpath)
    already_computed = []
    while True:
        ckpt_list = find_checkpoints(ckptpath)
        for c in ckpt_list:
            ckpt_epoch = int(re.findall(r'(\d{1,3})\.pt$', c)[0])
            if ckpt_epoch<FLAGS.resume_eval or c in already_computed:
                continue
            else:
                print("\nEvaluating", c)
                eval(0, ckpt_epoch, FLAGS, CIFAR10, 10)
                already_computed.append(c)
        time.sleep(100)


if __name__ == "__main__":
    print(f'PyTorch version: {torch.__version__}')
    app.run(main)
