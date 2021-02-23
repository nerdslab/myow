import torch
import tensorflow as tf
import tensorboard as tb

# fix a bug with tensorboard
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile


def log_representation(net, inputs, metadata, writer, step, tag='representation', metadata_header=None,
                       inputs_are_images=False):
    r"""
    Computes representations and logs them to tensorboard.

    Args:
        net (torch.nn.Module): Encoder.
        inputs (torch.Tensor): Inputs.
        writer (torch.writer.SummaryWriter): Summary writer.
        metadata (torch.Tensor or list): A list of labels, each element will be convert to string.
        step (int): Global step value to record.
        tag (string, optional): Name for the embedding. (default: :obj:`representation`)
        metadata_header (list, optional): Metadata header. (default: :obj:`None`)
        inputs_are_images (boolean, optional): Set to :obj:`True` if inputs are images. (default: :obj:`False`)
    """
    with torch.no_grad():
        representation = net(inputs)
        representation = representation.view(representation.shape[0], -1).detach()

    label_img = inputs if inputs_are_images else None
    writer.add_embedding(representation, metadata, tag=tag, global_step=step, metadata_header=metadata_header,
                         label_img=label_img)
