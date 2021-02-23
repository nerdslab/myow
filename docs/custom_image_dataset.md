# Using your own image datasets

Setting up your own dataset is straightforward. This code already contains an example of MYOW applied to the CIFAR-10  
dataset. We have made it easy to train MYOW on any dataset and here, we give a brief step-by-step guide to setting 
up your own.

Augmentations can either be done on the CPU or on the GPU. 

### On-GPU augmentations
When it is possible and desirable to perform augmentations on the GPU in batch mode, use the `transform` argument 
to specify the augmentations used during training to generate views. In this case, augmented view generation is 
fully handled by the trainer which only need to receive a batch of the original images. 

Any standard PyTorch dataset can be used out of the box. A function called `prepare_views` needs to be defined, this 
function handles the definition of a dictionary that contains the two views needed by the trainer. In this case, 
the image is given twice.

```python
def prepare_views(inputs):
    x, labels = inputs
    outputs = {'view1': x, 'view2': x}
    return outputs
```

When constructing the trainer, simply specify:

```python
MYOWTrainer(..., prepare_views=prepare_views, transform=transform)
```

In some situations, it might be desired to define two different classes of transformations, this can be done by 
separately defining `transform_1` and `transform_2` with each being applied to their respective view.

```python
MYOWTrainer(..., prepare_views=prepare_views, transform_1=transform_1, transform_2=transform_2)
```

**View mining**

A set of candidates needs to be generated from the dataset during mining. When working with an image dataset, this can 
easily be done through a second dataloader passed to the trainer through `view_pool_dataloader`. The batch size of this
dataloader can be different from the batch size of the main dataloader. The transformation used during mining is 
specified through `transform_m` which will be applied to the key sample as well as all candidate samples.

```python
MYOWTrainer(..., view_pool_dataloader=view_pool_dataloader, transform_m=transform_m)
```

### On-CPU augmentations

To perform augmentation on the CPU in single image mode, the augmented views need to be generated before being passed
to the trainer. In this case, a custom Dataset object can be implemented along with `prepare_views` which
is used to assign the views. 

```python
def prepare_views(inputs):
    x1, x2 = inputs
    outputs = {'view1': x1, 'view2': x2}
    return outputs
```

When constructing the trainer, there is no need to specify the `transform`:

```python
MYOWTrainer(..., prepare_views=prepare_views, transform=None)
```

**View mining**

Similarly, the batch of candidate views needs to be generated pre-`trainer`. The custom Dataset needs to generate 4 
tensors, given an input `x`: two augmented views of `x` using the main class of transformations, one augmented view
using `transform_m` and a batch of candidate views. 

```python
def prepare_views(inputs):
    x1, x2, x3, xpool = inputs
    outputs = {'view1': x1, 'view2': x2, 'view3':x3, 'view_pool':xpool}
    return outputs
```
