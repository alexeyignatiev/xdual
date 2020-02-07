# XDual

A set of Python scripts for explaining boosted tree models by computing either abductive or contrastive explanations (or both), based on the hitting set duality between the two concepts. The implementation targets tree ensembles trained with [XGBoost](https://xgboost.ai/) and supports computing and enumerating subset- and cardinality-minimal *rigorous* explanations.

## Getting Started

Before using XDual, make sure you have the following Python packages installed:

* [anytree](https://anytree.readthedocs.io/)
* [numpy](http://www.numpy.org/)
* [pandas](https://pandas.pydata.org/)
* [pySMT](https://github.com/pysmt/pysmt)
* [PySAT](https://github.com/pysathq/pysat)
* [scikit-learn](https://scikit-learn.org/stable/)
* [XGBoost](https://github.com/dmlc/xgboost/)

Please, follow the installation instructions on these projects' websites to install them properly. (If you spot any other package dependency not listed here, please, let us know.)

## Usage

XDual has a number of parameters, which can be set from the command line. To see the list of options, run:

```
$ xdual.py -h
```

### Preparing a dataset

XDual can be used with datasets in the CSV format. If a dataset contains continuous data, you can use XDual straight away (with no option ```-c``` specified). Otherwise, you need to process the categorical features of the dataset. For this, you need to do a few steps:

1. Assume your dataset is stored in file ```somepath/dataset.csv```.
2. Create another file named ```somepath/dataset.csv.catcol``` that contains the indices of the categorical columns of ```somepath/dataset.csv```. For instance, if columns ```0```, ```1```, and ```5``` contain categorical data, the file should contain the lines

	```
	0
	1
	5
	```

3. Now, the following command:

```
$ xdual.py -p --pfiles dataset.csv,somename somepath/
```

creates a new file ```somepath/somename_data.csv``` with the categorical features properly handled. As an example, you may want to check the command on the [benchmark datasets](bench), e.g.

```
$ xdual.py -p --pfiles compas.csv,compas bench/fairml/compas/
```

### Training a tree ensemble

Before extracting explanations, an XGBoost model must be trained:

```
$ xdual.py -c -t -n 50 bench/fairml/compas/compas_data.csv
```

Here, 50 trees per class are trained. Also, parameter ```-c``` is used because the data is categorical. By default, the trained model is saved in the file ```temp/compas_data/compas_data_nbestim_50_maxdepth_3_testsplit_0.2.mod.pkl```.

### Computing an abductive explanation

Assuming that one targets explaining a data instance `5,0,0,0,0,0,0,0,0,0,0`, a rigorous *abductive* explanation for such an instance can be computed by running the following command:

```
$ xdual.py -c -e smt -s z3 -x '5,0,0,0,0,0,0,0,0,0,0' -v temp/compas_data/compas_data_nbestim_50_maxdepth_3_testsplit_0.2.mod.pkl
```

Here, parameter ```-e``` specifies the model encoding (SMT) while parameter ```-s``` identifies an SMT solver to use (various SMT solvers can be installed in [pySMT](https://github.com/pysmt/pysmt) - here we use [Z3](https://github.com/Z3Prover/z3)). This command computes a *subset-minimal* explanation, i.e. it is guaranteed that *no proper subset* of the reported explanation can serve as an explanation for the given prediction.

Alternatively, a *cardinality-minimal* (i.e. smallest size) explanation can be computed by specifying the ```-M``` option additionally:

```
$ xdual.py -c -e smt -M -s z3 -x '5,0,0,0,0,0,0,0,0,0,0' -v temp/compas_data/compas_data_nbestim_50_maxdepth_3_testsplit_0.2.mod.pkl
```

### Computing a contrastive explanation

Similarly, one can compute either subset- or cardinality-minimal rigorous *contrastive* explanation. For example, a rigorous *contrastive* explanation for the same data instance can be computed by running the following command:

```
$ xdual.py -c -e smt -s z3 -x '5,0,0,0,0,0,0,0,0,0,0' --xtype contrastive -v temp/compas_data/compas_data_nbestim_50_maxdepth_3_testsplit_0.2.mod.pkl
```

Alternatively, a cardinality-minimal contrastive explanation can be computed by specifying the ```-M``` option additionally:

```
$ xdual.py -c -e smt -M -s z3 -x '5,0,0,0,0,0,0,0,0,0,0' --xtype contrastive  -v temp/compas_data/compas_data_nbestim_50_maxdepth_3_testsplit_0.2.mod.pkl
```

### Enumeration of explanations

XDual also supports enumeration of either abductive or contrastive explanations, or both. The tool can be instructed to enumerate explanations by specifying the number of explanations to compute, which can be done using the ```-N``` option:

```
$ ./xdual.py -N 2 -c -e smt -M -s z3 -x '5,0,0,0,0,0,0,0,0,0,0' --xtype contrastive  -v temp/compas_data/compas_data_nbestim_50_maxdepth_3_testsplit_0.2.mod.pkl
```

Running this command would compute two contrastive explanations. If instead of an integer, a user puts `-N all`, XDual will enumerate all contrastive explanations for a given data instance.

The same command-line option should be used for enumerating abductive explanations. **Note** that for abductive explanations, the value of option `--xtype` should be set to `abductive` (or unset).

Combinations of options can be used to choose the explanation enumeration algorithm and its additional parameters, including options `-M`, `-u`,`--use-cld`, and `--use-mhs`. (For the combination used in the experiment described in the paper, pleasee, see the [experiment](experiment) directory.)

## Reproducing experimental results

Although it seems unlikely that the experimental results reported in the paper can be reproduced (due to *randomization* used in the training phase), similar results can be obtained if the following commands are executed:

```
$ cd experiment/
$ ./train-all.sh && ./extract-samples.sh
$ ./enumerate-all.sh
```

The final command should run the experiment the way it was set up for the paper. (**Note** that this will take a while.) The result files will contain the necessary statistics.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
