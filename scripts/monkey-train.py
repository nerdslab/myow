import os
import copy

from absl import app
from absl import flags
import torch
import tensorflow as tf
from torch.utils.data import DataLoader
from tqdm import tqdm

from self_supervised.data import ReachNeuralDataset, get_angular_data
from self_supervised.data import generators, utils
from self_supervised.transforms import neural_transforms as transforms
from self_supervised.trainer import MYOWTrainer
from self_supervised.nets import MLP
from self_supervised.tasks import neural_tasks
from self_supervised.utils import set_random_seeds


FLAGS = flags.FLAGS

# Dataset
flags.DEFINE_string('data_path', './data/mihi-chewie', 'Path to monkey data.')
flags.DEFINE_enum('primate', 'chewie', ['chewie', 'mihi'], 'Primate name.')
flags.DEFINE_integer('day', 1, 'Day of recording.', lower_bound=1, upper_bound=2)
flags.DEFINE_float('train_split', 0.8, 'train/test split', lower_bound=0., upper_bound=0.99)

# Transforms
flags.DEFINE_integer('max_lookahead', 5, 'Max lookahead.')
flags.DEFINE_float('noise_sigma', 0.2, 'Noise sigma.', lower_bound=0.)
flags.DEFINE_float('dropout_p', 0.8, 'Dropout probability.', lower_bound=0., upper_bound=1.)
flags.DEFINE_float('dropout_apply_p', 0.9, 'Probability of applying dropout.', lower_bound=0., upper_bound=1.)
flags.DEFINE_float('pepper_p', 0.0, 'Pepper probability.', lower_bound=0., upper_bound=1.)
flags.DEFINE_float('pepper_sigma', 0.3, 'Pepper sigma.', lower_bound=0.)
flags.DEFINE_float('pepper_apply_p', 0.0, 'Probability of applying pepper.', lower_bound=0., upper_bound=1.)
flags.DEFINE_boolean('structured_transform', True, 'Whether the transformations are consistent across temporal shift.')

# Dataloader
flags.DEFINE_integer('batch_size', 256, 'Batch size.')
flags.DEFINE_integer('pool_batch_size', 512, 'Batch size.')
flags.DEFINE_integer('num_workers', 4, 'Number of workers.')

# architecture
flags.DEFINE_integer('representation_size', 128, 'Representation size.')
flags.DEFINE_list('encoder_hidden_layers', [128, 128, 128], 'Sizes of hidden layers in encoder.')
flags.DEFINE_integer('projection_size', 32, 'Size of first projector.')
flags.DEFINE_integer('projection_hidden_size', 256, 'Size of hidden layer in first projector.')
flags.DEFINE_integer('projection_size_2', 16, 'Size of second projector.')
flags.DEFINE_integer('projection_hidden_size_2', 64, 'Size of hidden layer in second projector.')

# Training parameters
flags.DEFINE_float('lr', 0.8, 'Base learning rate.')
flags.DEFINE_float('mm', 0.9, 'Momentum for exponential moving average.')
flags.DEFINE_float('weight_decay', 1e-6, 'Weight decay.')
flags.DEFINE_float('myow_weight', 0.1, 'Base learning rate.')
flags.DEFINE_integer('miner_k', 3, 'k in knn during mining.')
flags.DEFINE_integer('num_epochs', 1000, 'Number of training epochs.')
flags.DEFINE_integer('lr_warmup_epochs', 10, 'Warmup period for learning rate.')
flags.DEFINE_integer('myow_warmup_epochs', 10, 'Warmup period during which mining is inactive.')
flags.DEFINE_integer('myow_rampup_epochs', 110, 'Rampup period for myow weight.')

# Random seed
flags.DEFINE_integer('random_seed', 100, 'Random seed.')


def main(argv):
    set_random_seeds(FLAGS.random_seed)

    # load dataset
    dataset = ReachNeuralDataset(FLAGS.data_path, primate=FLAGS.primate, day=FLAGS.day, binning_period=0.1,
                                 scale_firing_rates=False, train_split=FLAGS.train_split)
    dataset.train()
    firing_rates = dataset.firing_rates
    raw_labels = dataset.labels
    sequence_lengths = dataset.trial_lengths

    transform = transforms.Compose(transforms.RandomizedDropout(FLAGS.dropout_p, apply_p=FLAGS.dropout_apply_p),
                                   transforms.Normalize(torch.tensor(dataset.mean), torch.tensor(dataset.std)),
                                   transforms.Noise(FLAGS.noise_sigma),
                                   transforms.Pepper(FLAGS.pepper_p, FLAGS.pepper_sigma, apply_p=FLAGS.pepper_apply_p),
                                   )

    transform_val = transforms.Compose(transforms.Normalize(torch.tensor(dataset.mean), torch.tensor(dataset.std)),)

    pair_sets = utils.onlywithin_indices(sequence_lengths, k_min=-FLAGS.max_lookahead, k_max=FLAGS.max_lookahead)
    generator = generators.LocalGlobalGenerator(firing_rates, pair_sets, sequence_lengths,
                                                num_examples=firing_rates.shape[0],
                                                batch_size=FLAGS.batch_size,
                                                pool_batch_size=FLAGS.pool_batch_size,
                                                transform=transform, num_workers=FLAGS.num_workers,
                                                structured_transform=FLAGS.structured_transform)
    dataloader = DataLoader(generator, num_workers=FLAGS.num_workers, drop_last=True)

    # build encoder network
    input_size = firing_rates.shape[1]
    encoder = MLP([input_size, *FLAGS.encoder_hidden_layers, FLAGS.representation_size], batchnorm=True)

    trainer = MYOWTrainer(encoder=encoder,
                          representation_size=FLAGS.representation_size, projection_size=FLAGS.projection_size,
                          projection_hidden_size=FLAGS.projection_hidden_size,
                          projection_size_2=FLAGS.projection_size_2,
                          projection_hidden_size_2=FLAGS.projection_hidden_size_2,
                          base_lr=FLAGS.lr, base_momentum=FLAGS.mm, momentum=0.9, weight_decay=FLAGS.weight_decay,
                          optimizer_type='lars', batch_size=FLAGS.batch_size, total_epochs=FLAGS.num_epochs,
                          exclude_bias_and_bn=True, train_dataloader=dataloader, prepare_views=generator.prepare_views,
                          warmup_epochs=FLAGS.lr_warmup_epochs, myow_warmup_epochs=FLAGS.myow_warmup_epochs,
                          myow_rampup_epochs=FLAGS.myow_rampup_epochs, myow_max_weight=FLAGS.myow_weight,
                          view_miner_k=FLAGS.miner_k, gpu=0, log_step=10,
                          log_dir='runs-chewie1/myow_run_1')

    data_train, data_test = get_angular_data(dataset, device=trainer.device, velocity_threshold=5)

    def evaluate():
        trainer.model.eval()
        encoder_eval = copy.deepcopy(trainer.model.online_encoder)

        classifier = torch.nn.Sequential(torch.nn.Linear(FLAGS.representation_size, 2)).to(trainer.device)
        class_optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01, weight_decay=1e-5)

        acc, delta_acc = neural_tasks.train_angle_classifier(
            encoder_eval, classifier, data_train, data_test, class_optimizer,
            transform=transform, transform_val=transform_val, device=trainer.device,
            num_epochs=100, batch_size=FLAGS.batch_size)

        trainer.writer.add_scalar('trial_angles/acc_train', acc.train_smooth, trainer.step)
        trainer.writer.add_scalar('trial_angles/delta_acc_train', delta_acc.train_smooth, trainer.step)
        trainer.writer.add_scalar('trial_angles/acc_test', acc.val_smooth, trainer.step)
        trainer.writer.add_scalar('trial_angles/delta_acc_test', delta_acc.val_smooth, trainer.step)

    for epoch in tqdm(range(FLAGS.num_epochs + 1)):
        trainer.model.train()
        trainer.train_epoch()
        if epoch % 20 == 0:
            trainer.model.eval()
            evaluate()


if __name__ == "__main__":
    print(f'PyTorch version: {torch.__version__}')
    app.run(main)
