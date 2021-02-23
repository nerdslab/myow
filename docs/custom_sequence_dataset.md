# Using your own sequence datasets

This code already contains an example of MYOW applied to a dataset that contains a number of sequences. 
Here, we give a brief step-by-step guide to setting up your own sequence dataset.

The main augmentation is temporal shift, for that we use `data.generator.LocalGlobalGenerator` that takes as input the 
matrix of features as well as the list of possible pairs. This list is pre-computed to allow for faster data loading.
`data.utils.onlywithin_indices` can be used to generate such list.

```python
pair_sets = utils.onlywithin_indices(sequence_lengths, k_min=-2, k_max=2)
```

Then there is the generation of the pool of candidates for mining. Because we are working with time-varying data,
we need to restrict the candidate views used for mining to be separated by a minimum distance in time from the key 
views. We do this for our monkey datasets, by sampling two different sets of sequences from which we then we sample the 
key views and the pool of candidate views separately.

The additional transforms can be added through the `transform` argument.

```python
generator = generators.LocalGlobalGenerator(firing_rates, pair_sets, sequence_lengths,
                                            num_examples=firing_rates.shape[0],
                                            batch_size=batch_size,
                                            pool_batch_size=pool_batch_size,
                                            transform=transform, num_workers=1,
                                            structured_transform=True)
```

Similarly to images, the data needs to be specified for the trainer through the `prepare_views` function, which in this
case is defined as a static method of the generator. 

```python
@staticmethod
def prepare_views(inputs):
    x1, x2, x3, x4 = inputs
    outputs = {'view1': x1.squeeze(), 'view2': x2.squeeze(),
               'view3': x3.squeeze(), 'view_pool': x4.squeeze()}
    return outputs
```
