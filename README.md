# Sum-Product Autoencoding (SPAE)
Code and supplemental material for the work presented in: 

_Antonio Vergari, Robert Peharz, Nicola Di Mauro, Alejandro Molina, Kristian Kersting and Floriana Esposito_  
**Sum-Product Autoencoding: Encoding and Decoding Representations using Sum-Product Networks**  
in proceedings of of the Thirty-Two AAAI Conference on Artificial Intelligence (AAAI 2018)

## Supplemental material

The supplemental material containing detailed experimental settings and learners' training setups and performances, as well as proofs of the propositions provided in the original article, can be found in the `supplemental` directory as a pdf file.


## Code requirements

### Python packages

The code relies on the following `python3.4+` libs:

  * [numpy (1.10+)](https://www.numpy.org/),
  * [scipy (0.17+)](https://www.scipy.org/),
  * [sklearn (0.17+)](https://scikit-learn.org/stable/),
  * [pandas (0.15+)](https://pandas.pydata.org/),
  * [numba (0.27+)](https://numba.pydata.org/),
  * [pystruct (0.2.4+)](https://numba.pydata.org/) and [cvxopt (1.1.9)](http://cvxopt.org/) for structured CRF SVMs,
  * [matplotlib (2.0+)](https://matplotlib.org/) and [seaborn (0.6+)](https://seaborn.pydata.org/) for plots and visualizations,
  * [keras (2.0+)](http://http://keras.io/) not mandatory, for efficient inference on a layer-based GPU implementation of SPNs, leveraging [Theano (0.9+)](http://deeplearning.net/software/theano/) as a backend, but [tensorflow (1.0+)](https://www.tensorflow.org/) is equally good

### Numerical libraries (optional)
For optimal `theano`, `tensorflow` and `numpy` performances one can exploit the
`blas` and `lapack` libs. CUDA is required to exploit the GPU for inference.
To properly install them please refer to
[this page](http://deeplearning.net/software/theano/install.html)
The lib versions used on a Ubuntu 14.04 installation are:

```
liblapack3 3.5.0-2
libopenblas-dev 0.2.8-6
cuda 6.0.1
```

### Commands execution
[ipython (3.2+)](https://ipython.org/) has been used to launch all scripts.
The following commands will assume the use of `ipython` (`python3` interpreter is fine as well) and being in
the repo main directory:

```
cd spae
```

## Data
The 10 multilabel classification benchmark datasets used are freely accessible from the [MULAN](http://mulan.sourceforge.net/), [MEKA](http://meka.sourceforge.net/), and [LABIC](http://computer.njnu.edu.cn/Lab/LABIC/LABIC_Software.html) repositories.
They have been binarized, preprocessed as in [dcsn](https://github.com/nicoladimauro/dcsn) and divided into five folds.
They are provided here in the `data/multilabel` folder as compressed archives containing all five folds. E.g.,  for the dataset `arts1500` ("Arts" in the paper) the corresponding file is:

```
arts1500.5.folds.pklz
```

Furthermore,  the train and test splits of each fold are available separately (for debugging purposes). E.g.,

```
arts1500.0.train.data.gz
arts1500.0.test.data.gz
```

are the `train` and `test` splits for the first fold (`0`) for `arts1500`.

Additionally for the feature visualizations provided, the binarized version of MNIST employed can be found in the `data/bmnist.*.data` files and is also accessible from the [**spyn-repr** repo](https://github.com/arranger1044/spyn-repr) while the preprocessed, while the NIPS bag-of-words dataset can be found in [the **_Mixed Sum-Product Networks_** (**MSPNs**) repo](https://github.com/alejandromolinaml/MSPN).

## Learning Models
In the `models` directory, the SPN models employed in the
experiments over the **X** features only are provided as a compressed pickle archives: `<dataset-name>/best.<dataset-name>.5folds.pklz.<fold-id>.model.gz`.

Analogously, SPN models learned on the **Y** alone (for scenarios II and III of the MLC prediction experiments) can be found in the `models/y-only` sub-directory.

MADE models have been learned by employing the porting to [python3](https://github.com/arranger1044/MADE) of the original MADE implementation.

Lastly, all the autoencoder models involved have been learned by using [RAELk](https://github.com/arranger1044/RAELk) a little framework in `keras` to learn autoencoders for representation learning, please refer to that repo for training them.

The following sub sections will list the commands to learn the models back from data.

### Learning SPNs
To learn an SPN structure with `LearnSPN-b` one can use the script
`learnspn.py` in the `bin` directory, after specifying a dataset
name.

The G-Test threshold parameter values can be specified with `-g`.
The stopping criterion values for the min number of instances to split
is specified via the `-m` option.
The smoothing coefficient values can be specified through the `-a`
option.
We set the G-test independence test threshold to 5, we limit the
minimum number of instances in a slice to split to 10 and we performed 
a grid search for the best leaf distribution Laplace smoothing value in
`0.1, 0.2, 0.5, 1.0, 2.0`.


For instance, to learn a SPN model on the **X** features of the `arts1500` dataset, as used in the
paper, and saving its output in `exp/learnspn-repr-ml/` run the following command:


```
ipython -- bin/learnspn.py data/multilabel/arts1500/arts1500.5.folds.pklz  -k 2 -c GMM -g 5  -m 10 -a 0.1 0.2 0.5 1.0 2.0 -v 1 --cv 5 --save-model -o exp/learnspn-repr-ml/
```

Instead, to learn a model over the target **Y** variables, employ the `--y-only` option:

```
ipython -- bin/learnspn.py data/multilabel/arts1500/arts1500.5.folds.pklz  -k 2 -c GMM -g 5  -m 10 -a 0.1 0.2 0.5 1.0 2.0 -v 1 --cv 5 --y-only --save-model -o exp/learnspn-repr-ml/y-only/
```


Concerning the SPN model learned on the binarized MNIST employed in the embedding understanding section, you can decompress the model in `models/bmnist/bmnist_spn_50.tar.gz` or refer to the doc in [spyn-repr](https://github.com/arranger1044/spyn-repr) on how to re-learn it from data.

For the Poisson SPN used on the NIPS text data, please refer to the [MSPN repo](https://github.com/alejandromolinaml/MSPN).

## Extracting Embeddings
Given a dataset represented in the folds and compressed as illustrated before, over which a model has been learned in an unsupervised way, the embedding
generation functions will produce new representations for the corresponding folds according to the model provided and some filtering criterion.
Generating embeddings from SPN models, in particular, will produce also a _feature file map_ file,
comprising information about the node used to generate each feature.


### SPN embeddings
To extract both CAT and ACT embeddings for a dataset from a learned SPN model one can use
the `spn_repr.py` script in the `bin` sub-directory. 
One needs to specify the file path where to retrieve the original data as the
first parameter (e.g. `data/multilabel/arts1500/arts1500.5.folds.pklz`).
The SPN model path is specified with the option `--model` and the new
representation name will be composed using the `--suffix` parameter
value.

The output splits are generated in a pickle file, but can be saved in
the same format of the textual datasets with the `--save-txt` option (not recommended for large datasets).
To have these representation in a compressed archive use `--gzip`.

To specify how to extract the embeddings, one has to set two options:
`--ret-func` which determines which values to extract from a node (e.g., to
collect node output values in the log domain for ACT embedding use `"var-log-val"`), and
`--filter-func` that indicates which nodes to consider to generate the
embeddings (e.g., set it to `"all"` to get all nodes in a network).

Lastly, one can specify the numerical format of the input and output representation involved (forcing variable type casts) by employing the `--dtype` and `--repr-dtype` options respectively.

#### CAT embeddings

To extract CAT embeddings one has to look only at the latent (hidden) variables associated to sum nodes in an SPN.
Therefore, filtered nodes must be sum nodes `--filter-func "sum"` and the retrieval function for CAT latent variable assignment is `--ret-func "hid-cat"`. 

For the CAT embeddings as presented in the paper---which are inherently sparse---one has to look only at the sum nodes in the induced tree path by MaxProdMPE, which can by done by specifying `--sparsify-mpe -1` where `-1` is used as the numeric placeholder for discarded latent variable configurations.

For instance on the `arts1500` dataset the whole command is:

```
ipython -- bin/spn_repr.py data/multilabel/arts1500/arts1500.5.folds.pklz --model models/arts1500/best.arts1500.5.folds.pklz -o exp/ml-repr-x-cat-sparse/arts1500/ --ret-func "hid-cat" --filter-func "sum" --suffix "sum-hid-cat-sparse" --cv 5 --no-ext --dtype int32 --repr-dtype int32 --fmt float --gzip --sparsify-mpe -1
```

For the dense version of CAT embeddings---CAT-dense (see Supplemental Material)--- one can just run:

```
ipython -- bin/spn_repr.py data/multilabel/arts1500/arts1500.5.folds.pklz --model models/arts1500/best.arts1500.5.folds.pklz -o exp/ml-repr-x-cat/arts1500/ --ret-func "hid-cat" --filter-func "sum" --suffix "sum-hid-cat" --cv 5 --no-ext --dtype int32 --repr-dtype int32 --fmt float --gzip
```

#### ACT embeddings

To collect activations of an SPN into a continuous embedding, one can specify `--ret-func "var-log-val"` as the retrieval function.
To obtain the ACT embeddings employed in the paper, we shall collect only inner node activations, i.e., discarding leaves, by using `--filter-func "non-lea"`, after turning an SPN into an MPN with the `--mpn` option (in this case for embedding the **Y** variables):

```
ipython -- bin/spn_repr.py data/multilabel/arts1500/arts1500.5.folds.pklz --model models/y-only/arts1500/best.arts1500.5.folds --y-only -o exp/ml-repr-data-y-no-leaf/arts1500/ --ret-func "var-log-val" --filter-func "non-lea" --suffix "y-non-leaf-log-val" --cv 5 --no-ext --mpn --dtype int32 --repr-dtype float --fmt float --gzip
```

If, on the other hand, one wants to collect activations from all nodes in the network---ACT-full embeddings (see Supplemental Material)---the filtering option `--filter-func "all"` shall be specified:

```
ipython -- bin/spn_repr.py data/multilabel/arts1500/arts1500.5.folds.pklz --model models/y-only/arts1500/best.arts1500.5.folds --y-only -o exp/ml-repr-data-y/arts1500/ --ret-func "var-log-val" --filter-func "all" --suffix "y-all-log-val" --cv 5 --no-ext --mpn --dtype int32 --repr-dtype float --fmt float --gzip
```


## Visualizations
Use the `visualize_spn.py` script to reproduce the visualizations
pictured in the paper. To save the outputs use the options: `--save
pdf -o <output-path>`.
The following commands show how to set some parameters to reproduce
the figures and plot shown in the paper.

Some common options among the commands are: `--size N M` to specify
how to render the images as matrices (for the binarized MNIST `M` = `N` = 28 by default). 
`--n-cols K` determines to display the
images in a grid of `K` columns.
`--max-n-images` limits the output of a command to a number of images.
`--invert` inverts the black and
white in the displayed images; `--space` determines the horizontal and
vertical space among displayed images, when in a grid.

### Visualizing mpe filters
To visualize the learned filters for the nodes by exploiting MPE
inference, use the `--mpe` option.
To reproduce a visualization by scope length used in the paper, run
a command like this one:

```
ipython -- bin/visualize_spn.py bmnist --model models/bmnist/bmnist_spn_50/best.bmnist.model --mpe scope --scope-range 10 100 --invert --n-cols 3 --max-n-images 9 -o exp/vis-mpe/
```

where `--mpe scope` determines the visualization by scope length and
`--scope-range` actually specify a range of scope lengths.

### Visualizing collapsed MPE clusters

To visualize the groups of instances with the same MPE induced tree
path, use the option `--hid-groups`, like in this way:

```
ipython -- bin/visualize_spn.py bmnist --hid-groups /media/valerio/formalità/repr/bmnist/mpe/500-mpe-hid-var.bmnist.pickle -1 --size 28 28 --n-cols 3 --max-n-images 9
```

when `--hid-groups` specifies the path to a pickle containing the
a CAT embedding representation of MNIST splits, as extracted by `spn_repr.py` (see [Extracting CAT embeddings](# CAT-embeddings)).





## MLC reconstruction tasks

To measure the reconstruction performances of ACT embeddings (CAT ones are equivalent for this task, see Propositions in the paper), use the `mpn_decode_encode_score.py` script as provided in the `bin` dir.

One has to specify a `dataset` and an SPN model via the `--model` option as well as the MLC metrics to employ.
The option `--no-leaves` specifies ACT embeddings as used in the paper (discarding leaf information) while, by default, ACT-full embeddings will be extracted and checked (see Supplemental material for encoding and decoding with ACT-full embeddings).

For instance, on `arts1500`, one can measure the `jaccard`, `hamming`, and `exact` match reconstruction scores by running:

```
ipython -- bin/mpn_decode_encode_score.py data/multilabel/arts1500/arts1500.5.folds.pklz --cv 5 --model models/arts1500/best.arts1500.5.folds.pklz --scores jaccard hamming exact --no-leaves
```

For models learned on the **Y** target variables, use the `--y-only` option.

## MLC predition tasks

To employ the learned representation in multi-label classification task, use
the `enc_dec_classify_repr.py` script. It takes as arguments the name of
the dataset and other parameters to determine the classifier
configuration and specify a grid search over the remaining base classifier hyperparameters.
In case of scenarios II and III, in which a decoder SPN is required, the corresponding model path can be specified by the option `--decode-model`.
The option `--emb-type` can be either `activations` or `latent_categorical` and is used for ACT, resp. CAT embeddings.

Moreover, the representations for the input **X** features shall be specified by the `--repr-x` option while the ones for the output **Y** with `--repr-y` (`--x-orig` can be used to employ the original dataset features as the representations for the input variables).
The options `--repr-x-dtype` and `--repr-y-dtype` specify the types for the input and output representations employed.

To reproduce the experiments in the paper use as the base  classifier the L2-regularized logistic regressor (LR)  for classification by specifying `--classifier "lr-l2-mlo-bal"` while a Ridge regressor (RR) for regression by specifying `--classifier "rr-l2-ovr-bal"`. Regularization hyperparameters for both methods can be evaluated in a grid by using the option `--log-c`.

The available scores to evaluate the prediction performances are `jaccard hamming exact micro-f1 macro-f1 micro-auc-pr macro-auc-pr`. With the `--save-preds` option one can dump the fold predictions into a serialized numpy array.

For instance, to train and evaluate an RR base model on ACT-full embeddings extracted from the labels **Y** `arts1500` for the scenario II (learn a predictor from original input **X** features to the new label embedding space on **Y**), run the command:

```
ipython -- bin/enc_dec_classify_repr.py data/multilabel/arts1500/arts1500.5.folds.pklz  --repr-x data/multilabel/arts1500/arts1500.5.folds.pklz --repr-y exp/ml-repr-data-y/arts1500/y-all-log-val.arts1500.5.folds.pklz --x-orig --decode-model models/y-only/arts1500/best.arts1500.5.folds --classifier "rr-l2-bal" --log-c 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 20.0 100.0 --dtype int32 --repr-x-dtype float --repr-y-dtype float --cv 5 --scores jaccard hamming exact -o exp/ml-repr-y-clf/ --save-preds
```

On the other hand, for CAT embeddings in the same scenario run a command like this:

```
ipython -- bin/enc_dec_classify_repr.py data/multilabel/arts1500/arts1500.5.folds.pklz  --repr-x data/multilabel/arts1500/arts1500.5.folds.pklz --repr-y exp/ml-repr-y-cat-sparse/arts1500/y-sum-hid-cat-sparse.arts1500.5.folds.pklz --x-orig --emb-type "latent_categorical" --decode-model models/y-only/arts1500/best.arts1500.5.folds --classifier "lr-l2-mlo-bal" --log-c 0.0001 0.001 0.01 0.1 1.0 --dtype int32 --repr-x-dtype int32 --repr-y-dtype int32 --cv 5 --scores jaccard hamming exact micro-f1 macro-f1 micro-auc-pr macro-auc-pr -o exp/ml-clf-cat-sparse/x-ey/
```

### k-NN decoding
To reproduce the  experiments employing  k-NN  as a decoding alternative for scenarios II and III, use the `--knn-decode "[n_neighbors=5,n_jobs=3]"` option which indicates to use a 5 nearest neighbor approach using 3 threads in parallel to speed up computations.

The following command specifies this option for the scenario II  employing ACT-full embeddings on `arts1500`:

```
ipython -- bin/enc_dec_classify_repr.py data/multilabel/arts1500/arts1500.5.folds.pklz  --repr-x data/multilabel/arts1500/arts1500.5.folds.pklz --repr-y exp/ml-repr-data-y/arts1500/y-all-log-val.arts1500.5.folds.pklz --x-orig --decode-model models/y-only/arts1500/best.arts1500.5.folds --classifier "rr-l2-bal" --log-c 0.0001 0.001 0.01 0.1 1.0 10.0 100.0 --dtype int32 --repr-x-dtype float --repr-y-dtype float --cv 5 --scores jaccard hamming exact --knn-decode "[n_neighbors=5,n_jobs=3]" -o exp/ml-repr-y-clf-knn/ --save-preds 
```

## MLC partial embedding evaluation

For the experiments evaluating partial embedding decoding when missing components are generated at random one has to use the `enc_dec_classify_repr.py` script with the same parameter options as specified above. Additionally, to specify different missing components percentages one can use the option `--missing-percs`.

For example, on ACT embeddings from `arts1500` on scenario II, one can run:

```
ipython -- bin/enc_dec_classify_repr.py data/multilabel/arts1500/arts1500.5.folds.pklz  --repr-x exp/ml-repr-data/arts1500/non-leaf-log-val.arts1500.5.folds.pklz --repr-y exp/ml-repr-data-y/arts1500/y-all-log-val.arts1500.5.folds.pklz --x-orig --decode-model models/y-only/arts1500/best.arts1500.5.folds --classifier "rr-l2-bal" --log-c 0.0001 0.001 0.01 0.1 1.0  10.0  100.0 --dtype int32 --repr-x-dtype float --repr-y-dtype float --cv 5 --scores jaccard hamming exact micro-f1 macro-f1 micro-auc-pr macro-auc-pr --missing-percs 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 -o exp/ml-repr-y-miss-clf/  
```
