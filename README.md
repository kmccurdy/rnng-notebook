# RNNG Notebook

## About

These notes and accompanying code were created as a presentation aid for the paper [Recurrent Neural Network Grammars](https://arxiv.org/abs/1602.07776), Dyer et al. 2016, at the [Berlin Machine Learning seminar](http://building-babylon.net/berlin-machine-learning-learning-group/). 

The code in `RNNG.py` is a reimplementation of Dyer et al. using Python bindings to [DyNet](https://dynet.readthedocs.io/en/latest/), and borrows heavily from two sources:

* The [original RNNG code](https://github.com/clab/rnng), implemented in C++
* The Python implementation of the stack LSTM parser from [Graham Neubig's NN4NLP course](https://github.com/neubig/nn4nlp-code)

Both are released under [the Apache license v2.0](http://www.apache.org/licenses/LICENSE-2.0), as is this work.

## Installing

Dependencies in this project are managed with [Pipenv](https://docs.pipenv.org/) - follow link to directions on how to install.

Once you have it, run to install dependencies:

```
pipenv install
```

Launch environment shell to run subsequent code steps:

```
pipenv shell
```


## Getting + preparing data

The data used for this notebook is the NLTK release of ~10% of the Penn Treebank ([Marcus et al., 1994](https://dl.acm.org/citation.cfm?id=1075835)). To get the data, download the file from the [NLTK data repo](https://github.com/nltk/nltk_data/blob/gh-pages/packages/corpora/treebank.zip) and unzip it in the directory `data` within this repo.

To get the treebank data in the necessary format and divide into train/dev/test sets, run:

```
python split_training_data.py
```

See source code to adjust filepaths, relative size of training / dev sets, etc.

To get the oracle data sets:

```
python get_oracle_gen.py data/train.ptb data/train.ptb > data/train.oracle
python get_oracle_gen.py data/train.ptb data/dev.ptb > data/dev.oracle
python get_oracle_gen.py data/train.ptb data/test.ptb > data/test.oracle
```

To get the Brown clusters used to support word generation (generated as described in [Koo et al. 2008](http://people.csail.mit.edu/maestro/papers/koo08acl.pdf)), [download them](http://people.csail.mit.edu/maestro/papers/bllip-clusters.gz) and unzip in the `data` directory.

## Notebook

Follow these [instructions](https://stackoverflow.com/questions/47295871/is-there-a-way-to-use-pipenv-with-jupyter-notebook) to install the jupyter notebook kernel within your virtual environment (after calling `pipenv shell`):

```
python -m ipykernel install --user --name=<rnng-notebook-[your local environment hash]>
```

Then launch the notebook server from within the shell:

```
jupyter notebook
```

Within the notebook, use the Kernel > Change kernel menu to use the kernel local to your virtual environment.


## License

This software is released under [the Apache license v2.0](http://www.apache.org/licenses/LICENSE-2.0).
