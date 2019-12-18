import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import os.path as op
import tempfile
import numpy as np
import pandas as pd
import pickle
from scipy.stats import spearmanr
import tensorflow as tf
from deepchem.models import GraphConvModel
from deepchem.feat import ConvMolFeaturizer
from deepchem.data import CSVLoader
from deepchem.metrics import Metric, r2_score
from compchemdl2.models import RESULTS_DIR
from compchemdl2.utils import ensure_dir


####################################################
# Defines how much GPU memory is allowed to be used
####################################################


config = tf.ConfigProto(
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75),
    device_count = {'GPU': 1}
)

###################
# Model definition
###################

def define_gc_regression_model(n_tasks, graph_conv_sizes=(128, 128), dense_size=256, batch_size=128,
                               learning_rate=0.001, config=config, model_dir='/tmp'):
    """
    Initializes the multitask regression GCNN
    :param n_tasks: number of output tasks
    :param graph_conv_sizes: tuple with output dimension for every GC layer
    :param dense_size: size of the dense layer
    :param batch_size: number of examples per minibatch
    :param learning_rate: initial learning rate
    :param config: GPU and memory usage options
    :param model_dir: where the trained model will be stored
    :return: a GraphConvModel object
    """

    return GraphConvModel(n_tasks=n_tasks, graph_conv_layers=graph_conv_sizes, dense_layer_size=dense_size,
                          dropout=0.0, mode='regression', number_atom_features=75, uncertainty=False,
                          batch_size=batch_size, learning_rate=learning_rate, learning_rate_decay_time=1000,
                          optimizer_type='adam', configproto=config, model_dir=model_dir)


#################################
# Data loading and featurization
#################################

def load_data(dataset_file, split_field=None, smiles_field='Smiles', y_field=None, id_field='Compound_No',
              tempdir=op.join(RESULTS_DIR, 'DCGraphConv', 'to_check'), cv=True):
    """
    :param dataset_file: path to the csv file containing the data to read
    :param split_field: column name in the csv giving the fold assignment for CV. Optional
    :param smiles_field: column name in the csv giving the SMILES of the compounds
    :param y_field: column name in the csv giving the experimental value to learn. Optional (in case of prospective
    predictions)
    :param id_field: column name in the csv giving the molecule id
    :param tempdir: where the shards and featurized data will be temporarily stored
    :param cv: bool. whether we are splitting by folds or not (in which case split_field is used as fold assignment
    indicator)
    :return: list of tasks and the list of ConvMol datasets (one dataset per group in split_field)
    """
    if y_field is None:
        tasks = []
    else:
        tasks = [y_field]
    featurizer = ConvMolFeaturizer()
    loader = CSVLoader(tasks=tasks, smiles_field=smiles_field, featurizer=featurizer, id_field=id_field)
    dataset = loader.featurize(dataset_file, shard_size=8192, data_dir=tempdir)

    if split_field is not None and cv:
        df_train = pd.read_csv(dataset_file, sep=',')
        folds = np.unique(df_train[split_field].tolist())

        folds_datasets = []
        fold_dirs = []
        for f in folds:
            fold_dir = tempfile.mkdtemp(prefix=tempdir + '/')
            indices = np.flatnonzero(df_train[split_field] == f)
            folds_datasets.append(dataset.select(indices, select_dir=fold_dir))
            fold_dirs.append(fold_dir)

        return tasks, folds_datasets, fold_dirs

    return tasks, dataset


#####################
# Train and validate
#####################


def train_and_validate_mtnn(train, n_tasks, outdir, graph_conv_sizes, dense_size, batch_size, learning_rate, num_epochs,
                            pickle_file_name, test=None, test_unscaled=None, transformer=None, fold=None):
    """
    :param train: DeepChen dataset object, y appropriately scaled already
    :param n_tasks: number of tasks in the data
    :param outdir: where to store the outputs
    :param batch_size: number of examples per minibatch
    :param learning_rate: initial learning rate
    :param graph_conv_sizes: tuple with output dimension for every GC layer
    :param dense_size: size of the dense layer
    :param num_epochs: number of epochs to perform training
    :param pickle_file_name: how to call the file that will contain ytrain, yhattrain, etc.
    :param test: optional. Can be a DeepChem dataset object in case we want to validate the thing, with y already scaled
    as needed. If not, only training set fitting will be monitored.
    :param test_unscaled: optional. Can be a DeepChem dataset object with y as in the original dataset.
    :param transformer: optional. transformer object used to transform train and test (normally, z-scaler for the y).
    :param fold: fold number in case we are doing CV. Will be used as a suffix for pickle files
    :return: y_true, y_pred, and weights for the training (and also for test if provided)
    """

    # 1. Define the model
    model_dir = op.join(outdir, 'model')
    ensure_dir(model_dir)
    model = define_gc_regression_model(n_tasks, graph_conv_sizes=graph_conv_sizes,
                                       dense_size=dense_size, batch_size=batch_size,
                                       learning_rate=learning_rate, model_dir=model_dir)

    # 2. Define the metrics
    r2 = Metric(r2_score, np.mean)
    spearman = Metric(spearmanr, np.mean, mode='regression', name='spearman_rho')

    # 3. Train the model
    for l in range(0, num_epochs):
        print('EPOCH %i' % l)
        model.fit(train, nb_epoch=1)  # at every epoch, stop and evaluate on training set
        model.evaluate(train, [r2, spearman])
        if test is not None:
            try:
                model.evaluate(test, [r2, spearman])
            except TypeError:  # nan in one of the tasks, for ex
                print('No validation performance available')

    # 4. Obtain final performance
    yhattrain = model.predict(train)
    if test is not None:
        yhattest = model.predict(test)

    # 5. Save the model and the predictions
    print('Saving results...')
    # save the ys in a pickle file (also include the non scaled y for test set so we can revert back and compare)
    with open(op.join(outdir, pickle_file_name), 'wb') as writer:
        if test is not None:
            pickle.dump([train.y, yhattrain, train.w, test.y, yhattest, test.w, test_unscaled.y], writer,
                        protocol=pickle.HIGHEST_PROTOCOL)
        else:
            pickle.dump([train.y, yhattrain, train.w], writer, protocol=pickle.HIGHEST_PROTOCOL)

    model.save()

    # save the transformer for inference time...
    if fold is not None:
        transformer_file = op.join(outdir, 'transformer_fold_%i.pkl' % fold)
        zscale_file = op.join(outdir, 'transformer_fold_%i_easyread.pkl' % fold)
    else:
        transformer_file = op.join(outdir, 'transformer.pkl')
        zscale_file = op.join(outdir, 'transformer_easyread.pkl')
    with open(transformer_file, 'wb') as writer:
        pickle.dump(transformer, writer, protocol=pickle.HIGHEST_PROTOCOL)
    with open(zscale_file, 'wb') as writer:
        pickle.dump([transformer.y_means, transformer.y_stds], writer, protocol=pickle.HIGHEST_PROTOCOL)

    # save the molids...
    if fold is not None:
        molids_file = op.join(outdir, 'molids_fold_%i.pkl' % fold)
    else:
        molids_file = op.join(outdir, 'molids.pkl')
    with open(molids_file, 'wb') as writer:
        pickle.dump([train.ids, test.ids], writer, protocol=pickle.HIGHEST_PROTOCOL)

    # Signal that this training is over by creating an empty DONE.txt file
    open(op.join(outdir, 'DONE.txt'), 'a').close()

    if test is not None:
        return train.y, yhattrain, train.w, test.y, yhattest, test.w
    else:
        return train.y, yhattrain, train.w
