# MTNNGC_ADMET

This project contains the code to train / validate / predict a multi-task graph convolutional model using DeepChem. The architecture is the one used in the paper "Modeling Physico-Chemical ADMET Endpoints With Multitask Graph Convolutional Networks".


## Setting up an environment

The conda environment containing all necessary dependencies can be created using the environment.yml file as so:

```conda env create -f environment.yml```

The environment should then be activated: ```source activate mtnngc```

And finally the compchemdl library should be installed:

```pip install -e .```

This is to be run once, then the environment should simply be activated as explained above before trying to run any of the functionalities of the library.

## Training and validating models

MTNNGC models can be trained using the python script in **compchemdl/models/mtnn_gc.py**. It allows to run cross-validation, train-test, or simple train of the model. The options are the following:


- --tasks (-t): this should be the path to a directory where your training sets are stored (one training set per task in the folder). You can used the directory we prepared with public data in **data/training_sets** for example.

- --test_tasks (-tt): *Optional*. This is the path to the directory where the test sets are stored (one test set per task in the folder, should have the same names as the corresponding training set files in the folder indicated at the --tasks option.

- --name (-n): how the model should be named (ex: ADMET_1)

- --output (-o): *Optional*. How the temporary directory storing intermediate results should be called

- --cv (-x): Whether to run a cross-validation on the training sets or not. *Default: True* NB: setting this to true requires that the input files contain a column with cross-validation fold assignments

- --refit (-r): Whether to run a final training on the whole training sets. *Default: False*

- --batch (-b): size of the minibatches. *Default: 128*

- --epochs (-e): number of training epochs. *Default: 40*

- --learningrate (-l): learning rate. *Default: 0.001*

- --gpu (-g): *Optional* GPU to use. If nothing is passed, the CPU will be used instead.

- --smiles_field (-s): header of the column containing the clean SMILES of the input molecules. *Default: 'canonical_smiles'*

- --y_field (-y): header of the column containing the target value for every task. *Default: 'label'*

- --id_field (-i): header of the column containing the identifiers of the input molecules. *Default: 'mol_index'*

- --split_field (-f): header of the column containing the fold assignments for cross-validation. Only needed if --cv is set to True. *Default: 'fold'*

We provide in this repository a folder with three example datasets that can be used to test. The datasets do not correspond to the ones discussed in the paper but rather open data coming from ChEMBL. 
To try and run a cross-validation task, the following python command can be used (if your GPU 0 is available):

```python -u compchemdl/models/mtnn_gc.py -t /your_path_to_the_repo/data/training_data -n test_model -o mytemp -x True -r False -g 0```

To train the same model on the whole data and save it for future use, the following python command can be used:

```python -u compchemdl/models/mtnn_gc.py -t /your_path_to_the_repo/data/training_data -n test_model -o mytemp -x False -r True -g 0```


## Dataset preparation

## Inference
