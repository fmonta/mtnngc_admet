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


- 


## Dataset preparation

## Inference
