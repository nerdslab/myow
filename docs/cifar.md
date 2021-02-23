# Running MYOW on CIFAR10 dataset

This page walks through the steps required to run MYOW on the CIFAR10 dataset.

## Running training
Training is parallalised using `DistributedDataParallel`. The pool of candidate views during mining is shared across
all instances.

To start training run:

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 scripts/cifar-train.py \
    --lr 2.0 \
    --mm 0.98 \
    --weight_decay 5e-5 \
    --optimizer sgd \
    --lr_warmup_epochs 30 \
    --batch_size 256 \
    --port 12354 \
    --logdir myow-cifar \
    --ckptpath myow-cifar
```

## Running evaluation

Evaluation can be done simultaneously or after training on a separate GPU instance. The eval script will automatically 
run evaluation each time a new checkpoint is saved to `ckptpath`. It is also possible to start evaluation only after
a certain number of epoch using the `resume_eval` argument.

```bash
CUDA_VISIBLE_DEVICES=2 python3 scripts/cifar-eval.py \
    --lr 0.04 \
    --resume_eval 0 \
    --logdir runs-cifar \
    --ckptpath myow-cifar
```

## Running tensorboard

```bash
tensorboard --logdir=runs-cifar
```
