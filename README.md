# MYOW

PyTorch implementation of 
[Mine Your Own vieW: Self-Supervised Learning Through Across-Sample Prediction](https://arxiv.org/abs/2102.10106).

## Installation
To install requirements run:
```bash
python3 -m pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:$PWD
```


## Training
* <a href='docs/cifar.md'>Running MYOW on CIFAR-10 dataset.</a><br>
* <a href='docs/monkey_reach.md'>Running MYOW on neural recordings from primates.</a><br>

Setting up your own datasets:

* <a href='docs/custom_image_dataset.md'>Image dataset.</a><br>
* <a href='docs/custom_image_dataset.md'>Temporal dataset.</a><br>

## Contributors

*   Mehdi Azabou (Maintainer), github: [mazabou](https://github.com/mazabou)
*   Ran Liu, github: [ranliu98](https://github.com/ranliu98)
*   Kiran Bhaskaran-Nair, github: [kbn-gh](https://github.com/kbn-gh)
*   Erik C. Johnson, github: [erikjohnson24](https://github.com/erikjohnson24)

## Citation
If you find the code useful for your research, please consider citing our work:

```
@misc{azabou2021view,
      title={Mine Your Own vieW: Self-Supervised Learning Through Across-Sample Prediction}, 
      author={Mehdi Azabou and Mohammad Gheshlaghi Azar and Ran Liu and Chi-Heng Lin and Erik C. Johnson 
              and Kiran Bhaskaran-Nair and Max Dabagia and Keith B. Hengen and William Gray-Roncal 
              and Michal Valko and Eva L. Dyer},
      year={2021},
      eprint={2102.10106},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

