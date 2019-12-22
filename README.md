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

### Training and validation 
The datasets for training and testing should be comma-separated files with different fields: a field for the identifier of the molecule (*id_field*, set by default to 'mol_index'), a field for the structures in SMILES format (*smiles_field*, set by default to 'canonical_smiles'), a field for the target value of the particular task (*y_field*, set by default to 'label'). In case cross-validation is required, then an additional field to assign every example to a CV fold is needed (*split_field*, set by default to 'fold'). Example files can be found under the data/training_data folder.
One file per task is required, and will be aggregated based on the SMILES column, hence it is necessary to preprocess the compounds the same way for all tasks. 

When train-test is required, a second directory containing test files should be given. The format is exactly the same as for the training and cross-validation. It is important to give the same names to the files in both directories so that they are matched properly (ex: Task1_train.csv, Task2_train.csv in training_data/ and Task1_test.csv, Task2_test.csv in the test directory). 

### Inference

For inference, we require an .smi format as input: tab separated file with one row per compound, starting with the SMILES in the first column and a molecule identifier in the second column. No headers.
An example file can be found under data/test.smi.


## Inference

Once a model has been trained and validated, it can also be used for predicting new compounds. For this, we provide the script in **compchemdl/inference/inference.py**. The options are the following:

- --input (-i): path to the input file in .smi format (see Dataset preparation for more details)
- --checkpoint (-c): path to the directory where the final trained model is stored
- --output (-o): *Optional* path to the csv file where the predictions can be stored. If no path is given, the predictions will be printed out instead.
- --jobdir (-j): temporary directory where intermediate files are to be stored. Will be deleted at the end of the job
- --gpu (-g): *Optional* GPU to use. If nothing is passed, the CPU will be used instead.

Assuming the model named "model_1" has been previously trained and saved on date XX-XX-XX, using it for prediction on our test file would be done the following way:

```python -u compchemdl/inference/inference.py -i /*your_path_to_the_repo*/data/test.smi -c /*your_path_to_the_repo*/data/models/model_1/final_model_XX-XX-XX -j /tmp```
