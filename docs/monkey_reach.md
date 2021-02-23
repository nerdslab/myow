# Running MYOW on Reach dataset

This page walks through the steps required to run MYOW on the monkey datasets.

## Dataset
Processed data can be found in `data/`

## Running training and evaluation

Training can be run using the following command:

```bash
python3 scripts/monkey-train.py \
    --data_path=./data/mihi-chewie \
    --primate="chewie" \
    --day=1 \
    --max_lookahead=4 \
    --noise_sigma=0.1 \
    --dropout_p=0.8 \
    --dropout_apply_p=0.9 \
    --structured_transform=True \
    --batch_size=256 \
    --pool_batch_size=512 \
    --miner_k=3 \
    --myow_warmup_epochs=10 \
    --myow_rampup_epochs=110 
```
where `primate` and `day` specify the animal.


## Running tensorboard

```bash
tensorboard --logdir=runs-chewie1
```