import os
import os.path as op
import pandas as pd
import tempfile
import shutil
import pickle
from collections import defaultdict

from deepchem.data import DiskDataset

from compchemdl.data.datasets import load_training_data
from compchemdl.data import MODEL_DIR as RESULTS_DIR
from compchemdl.utils import ensure_dir, evaluate, str2bool
from compchemdl.models.deepchem_transformers import NormalizationTransformer, undo_transforms
from compchemdl.models.deepchem_graphconv import train_and_validate_mtnn

####################
# OUTPUT DIRECTORIES
####################

def get_multitask_cv_outdir(tasks_nickname, fold_number):
    return op.join(RESULTS_DIR, tasks_nickname, 'fold_%i' % fold_number)


def get_multitask_traintest_outdir(tasks_nickname):
    return op.join(RESULTS_DIR, tasks_nickname, 'train_test')


def get_multitask_outdir(tasks_nickname):
    import datetime
    now = datetime.datetime.now()
    return op.join(RESULTS_DIR, tasks_nickname, 'final_model_' + now.strftime('%Y-%m-%d'))


################
# Train/test
################

def train_test_mtnn(train_task_csvs, test_tasks_csvs, tasks_nickname, smiles_field, y_field, id_field,
                    tempdir, num_epochs=40, batch_size=128, learning_rate=0.001, graph_conv_sizes=(128, 128),
                    dense_size=256, gpu=None):
    """
    Trains a multitask GCNN using the training sets in train_tasks_csvs and validates it using the test sets in
    test_tasks_csvs. Saves the trained model and the predictions under a folder named "train_test". Prints performance
    metrics (R2 and Spearman rho) after every epoch.
    NB: each task in the model should have a corresponding training and test files, named similarly (ex:task1_train.csv,
    task1_test.csv).
    :param train_task_csvs: list of csv files containing the training tasks
    :param test_tasks_csvs: list of csv files containing the test tasks
    :param tasks_nickname: how the model will be named
    :param smiles_field: in the csvs, name of the column containing the smiles string of the cpds
    :param y_field: in the csv, name of the column containing the activity data
    :param id_field: in the csv, name of the column containing the molids
    :param tempdir: where to store the temporary files for the DiskDatasets (will be deleted later on)
    :param num_epochs: how many epochs to train for
    :param batch_size: number of molecules per minibatch
    :param learning_rate: learning rate
    :param graph_conv_sizes: tuple with output dimension for every GC layer
    :param dense_size: nb of neurons in the last dense layer
    :param gpu: GPU to use for training (if None, only CPU will be used)
    :return: None
    """

    ensure_dir(tempdir)
    tasks, training_dset = load_training_data(train_task_csvs, smiles_field=smiles_field, y_field=y_field,
                                              id_field=id_field, tempdir=op.join(tempdir, 'train'), cv=False)
    tasks, test_dset = load_training_data(test_tasks_csvs, smiles_field=smiles_field, y_field=y_field,
                                          id_field=id_field, tempdir=op.join(tempdir, 'test'), cv=False)

    # Take care of outdir
    outdir = get_multitask_traintest_outdir(tasks_nickname)
    ensure_dir(outdir)

    # Have we already run that experiment?
    if op.exists(op.join(outdir, 'DONE.txt')):
        print('Model already trained and validated.')

    else:
        print('Training and validating multitask graph convolution model')

        # Merge to reduce the number of shards (very necessary to avoid weird problems of non-random minibatches)
        disk_dir_to_delete = tempfile.mkdtemp(prefix=tempdir + '/')
        training_dset = DiskDataset.merge([training_dset], merge_dir=disk_dir_to_delete)

        # Transformation (z-scaling)
        zscaling_dir_train = op.join(tempdir, 'zscaling', 'train')
        ensure_dir(zscaling_dir_train)
        zscaling_dir_test = op.join(tempdir, 'zscaling', 'test')
        ensure_dir(zscaling_dir_test)
        transfo_dir_to_delete_1 = tempfile.mkdtemp(prefix=zscaling_dir_train + '/')
        transfo_dir_to_delete_2 = tempfile.mkdtemp(prefix=zscaling_dir_test + '/')
        transformer = NormalizationTransformer(transform_y=True, dataset=training_dset)
        scaled_train = transformer.transform(training_dset, outdir=transfo_dir_to_delete_1)
        scaled_val = transformer.transform(test_dset, outdir=transfo_dir_to_delete_2)

        # Train the model
        scaled_train_y, yhattrain, scaled_train_w, scaled_test_y, yhattest, scaled_test_w = \
            train_and_validate_mtnn(scaled_train, n_tasks=len(tasks), outdir=outdir, graph_conv_sizes=graph_conv_sizes,
                                    dense_size=dense_size, batch_size=batch_size, learning_rate=learning_rate,
                                    num_epochs=num_epochs, pickle_file_name=tasks_nickname + '.pkl', test=scaled_val,
                                    transformer=transformer, test_unscaled=test_dset, gpu=gpu)

        # compute metrics
        scaled_results_test = evaluate_multitask_gc(scaled_test_y, yhattest, scaled_test_w)
        for k, vals in scaled_results_test.items():
            print(k)
            print(vals)

        # let's reverse the transformation from the predictions
        yhattest_untransf = undo_transforms(yhattest, [transformer])
        unscaled_results_test = evaluate_multitask_gc(test_dset.y, yhattest_untransf, test_dset.w)
        for k, vals in unscaled_results_test.items():
            print(k)
            print(vals)
        # hopefully the results are very similar

        # Remove transfo dir
        shutil.rmtree(transfo_dir_to_delete_1)
        shutil.rmtree(transfo_dir_to_delete_2)

    # Get rid of the temporary directory structure
    shutil.rmtree(tempdir)

    print('Dataset folders removed!')


#######
# CV
#######

def cross_validate_mtnn(task_csvs, tasks_nickname, smiles_field, split_field, y_field, id_field,
                        tempdir, num_epochs, batch_size=128, learning_rate=0.001, graph_conv_sizes=(128, 128),
                        dense_size=256, gpu=None):
    """
    Cross-validates a multitask GCNN using the training sets in train_tasks_csvs. Saves the trained models and the
    predictions under folders named "fold_i". Prints performance metrics (R2 and Spearman rho) after every epoch.
    NB: each task in the model should have a corresponding training file. A columns with fold assignment should be
    provided for the cross-validation.
    :param task_csvs: list of csv files containing the training tasks
    :param tasks_nickname: how the model will be named
    :param smiles_field: in the csvs, name of the column containing the smiles string of the cpds
    :param split_field: in the csvs, name of the column containing the fold assignment for the cross-validation
    :param y_field: in the csvs, name of the column containing the activity data
    :param id_field: in the csv, name of the column containing the molids
    :param tempdir: where to store the temporary files for the DiskDatasets (will be deleted later on)
    :param num_epochs: how many epochs to train for
    :param batch_size: number of molecules per minibatch
    :param learning_rate: learning rate
    :param graph_conv_sizes: tuple with output dimension for every GC layer
    :param dense_size: nb of neurons in the last dense layer
    :param gpu: GPU to use for training (if None, only CPU will be used)
    :return: A pandas dataframe with performance metrics for every fold
    """

    ensure_dir(tempdir)
    tasks, folds, fold_dirs = load_training_data(task_csvs, smiles_field=smiles_field, split_field=split_field,
                                                 y_field=y_field, id_field=id_field, tempdir=tempdir, cv=True)

    fold_results = defaultdict(list)

    for i, fold in enumerate(folds):

        # Take care of outdir
        outdir = get_multitask_cv_outdir(tasks_nickname, i)
        ensure_dir(outdir)

        # Have we already run that fold?
        if op.exists(op.join(outdir, 'DONE.txt')):
            print('Fold %i already computed.' %i)

        else:
            print('Running graph convolution model for fold %i' % i)
            val = fold
            disk_dir_to_delete = tempfile.mkdtemp(prefix=tempdir + '/')
            train = DiskDataset.merge(folds[0:i] + folds[i + 1:], merge_dir=disk_dir_to_delete)

            # Transformation (z-scaling)
            zscaling_dir_train = op.join(tempdir, 'zscaling', 'train')
            ensure_dir(zscaling_dir_train)
            zscaling_dir_test = op.join(tempdir, 'zscaling', 'test')
            ensure_dir(zscaling_dir_test)
            transfo_dir_to_delete_1 = tempfile.mkdtemp(prefix=zscaling_dir_train + '/')
            transfo_dir_to_delete_2 = tempfile.mkdtemp(prefix=zscaling_dir_test + '/')
            transformer = NormalizationTransformer(transform_y=True, dataset=train)
            scaled_train = transformer.transform(train, outdir=transfo_dir_to_delete_1)
            scaled_val = transformer.transform(val, outdir=transfo_dir_to_delete_2)

            train_y, yhattrain, train_w, test_y, yhattest, test_w = \
                train_and_validate_mtnn(scaled_train, len(tasks), outdir=outdir, graph_conv_sizes=graph_conv_sizes,
                                        dense_size=dense_size, batch_size=batch_size, learning_rate=learning_rate,
                                        num_epochs=num_epochs, pickle_file_name=tasks_nickname + '_fold_%i.pkl' % i,
                                        test=scaled_val, test_unscaled=val, transformer=transformer, fold=i, gpu=gpu)

            # compute metrics
            train_results = evaluate_multitask_gc(train_y, yhattrain, train_w)
            test_results = evaluate_multitask_gc(test_y, yhattest, test_w)

            # Populate the results dictionary
            for j, t in enumerate(tasks):
                fold_results['fold'].append(i)
                fold_results['task'].append(t)
                fold_results['train'].append(True)
                fold_results['r2'].append(train_results[j][0])
                fold_results['mse'].append(train_results[j][1])
                fold_results['mae'].append(train_results[j][2])
                fold_results['varex'].append(train_results[j][3])
                fold_results['spearman'].append(train_results[j][4])
                fold_results['fold'].append(i)
                fold_results['task'].append(t)
                fold_results['train'].append(False)
                fold_results['r2'].append(test_results[j][0])
                fold_results['mse'].append(test_results[j][1])
                fold_results['mae'].append(test_results[j][2])
                fold_results['varex'].append(test_results[j][3])
                fold_results['spearman'].append(test_results[j][4])

            # Clean the tempdirs
            shutil.rmtree(disk_dir_to_delete)
            shutil.rmtree(transfo_dir_to_delete_1)
            shutil.rmtree(transfo_dir_to_delete_2)
            print('folder removed!')

    # Get rid of the foldirs
    for foldir in fold_dirs:
        shutil.rmtree(foldir)
    shutil.rmtree(tempdir)
    print('fold dataset folders removed!')

    return pd.DataFrame.from_dict(fold_results)


###################
# Final training
###################

def train_multitask_gc(train_task_csvs, tasks_nickname, smiles_field, y_field, id_field, tempdir,
                       num_epochs, batch_size=128, learning_rate=0.001, graph_conv_sizes=(128, 128),
                       dense_size=256, gpu=None):
    """
    We assemble all the data we have on all tasks for a final training run.
    :param train_task_csvs: csv files of the training sets
    :param tasks_nickname: how to name the model (ex: 'PCtasks')
    :param smiles_field: in the csv, name of the column containing the smiles string of the cpds
    :param y_field: in the csv, name of the column containing the activity data
    :param id_field: in the csv, name of the column containing the molids
    :param tempdir: where to store the temporary files for the DiskDatasets (will be deleted later on)
    :param num_epochs: how many epochs to train for
    :param batch_size: number of molecules per minibatch
    :param learning_rate: learning rate
    :param graph_conv_sizes: tuple with output dimension for every GC layer
    :param dense_size: nb of neurons in the last dense layer
    :param gpu: GPU to use for training (if None, only CPU will be used)
    :return: None
    """
    ensure_dir(tempdir)

    # Get and merge the data
    tasks, training_dset = load_training_data(train_task_csvs, smiles_field=smiles_field, y_field=y_field,
                                              id_field=id_field, tempdir=op.join(tempdir, 'train'), cv=False)

    # Take care of outdir
    outdir = get_multitask_outdir(tasks_nickname)
    ensure_dir(outdir)

    # Have we already run that experiment?
    if op.exists(op.join(outdir, 'DONE.txt')):
        print('Model already trained and validated.')

    else:
        print('Training the final multitask graph convolution model')

        # Transformation (z-scaling)
        zscaling_dir_train = op.join(tempdir, 'zscaling')
        ensure_dir(zscaling_dir_train)
        transfo_dir_to_delete = tempfile.mkdtemp(prefix=zscaling_dir_train + '/')
        transformer = NormalizationTransformer(transform_y=True, dataset=training_dset)
        scaled_train = transformer.transform(training_dset, outdir=transfo_dir_to_delete)

        train_y, yhattrain, train_w = train_and_validate_mtnn(scaled_train, len(tasks), outdir,
                                                              graph_conv_sizes=graph_conv_sizes,
                                                              dense_size=dense_size, batch_size=batch_size,
                                                              learning_rate=learning_rate, num_epochs=num_epochs,
                                                              pickle_file_name=tasks_nickname + '.pkl', test=None,
                                                              test_unscaled=None, transformer=transformer, fold=None,
                                                              gpu=gpu)
        # compute metrics
        train_results = evaluate_multitask_gc(train_y, yhattrain, train_w)
        for k, vals in train_results.items():
            print(k)
            print(vals)

        # Remove temporary directory for transformer
        shutil.rmtree(transfo_dir_to_delete)

    # Get rid of the whole temporary directory structure
    shutil.rmtree(tempdir)

    print('Dataset folders removed!')


################
# Evaluation
################

def evaluate_multitask_gc_from_pickle(results_pkl):

    with open(results_pkl, 'rb') as reader:
        results = {}
        ytrain, yhattrain, ytrain_weights, ytest, yhattest, ytest_weights, ytest_original = pickle.load(reader)
        n_tasks = ytest.shape[1]

        for i in range(n_tasks):
            y = ytest[:, i]
            mask = ytest_weights[:, i]
            yhat = yhattest[:, i]
            results[i] = evaluate(y[mask], yhat[mask])

    return results


def evaluate_multitask_gc(ytrue, yhat, w):
    results = {}
    n_tasks = ytrue.shape[1]
    for i in range(n_tasks):
        y = ytrue[:, i]
        mask = w[:, i].astype(bool)
        yhat2 = yhat[:, i]
        results[i] = evaluate(y[mask], yhat2[mask])
    return results


#######################
# RUNNING JOB CONTROL
#######################

def dispatch_job(tasks_dir, model_nickname, tempdir_nickname, smiles_field='canonical_smiles',
                 y_field='label', id_field='mol_index', num_epochs=40, batch_size=128, learning_rate=0.001,
                 graph_conv_sizes=(128, 128), dense_size=256, split_field='fold', cv=True, final=False,
                 test_tasks_dir=None, gpu=None):

    if cv:
        print('Cross-validation run')
        files = sorted([op.join(tasks_dir, f) for f in os.listdir(tasks_dir) if op.isfile(op.join(tasks_dir, f))])
        return cross_validate_mtnn(files, model_nickname, split_field=split_field, smiles_field=smiles_field,
                                   y_field=y_field, id_field=id_field, tempdir=op.join(RESULTS_DIR, tempdir_nickname),
                                   num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate,
                                   graph_conv_sizes=graph_conv_sizes, dense_size=dense_size, gpu=gpu)

    else:
        if not final:
            print('Train-test case')
            assert test_tasks_dir is not None
            # train-test
            train_files = sorted([op.join(tasks_dir, f) for f in os.listdir(tasks_dir)
                                  if op.isfile(op.join(tasks_dir, f))])
            test_files = sorted([op.join(test_tasks_dir, f) for f in os.listdir(test_tasks_dir)
                                 if op.isfile(op.join(test_tasks_dir, f))])

            return train_test_mtnn(train_files, test_files, model_nickname, smiles_field=smiles_field,
                                   y_field=y_field, id_field=id_field, tempdir=op.join(RESULTS_DIR, tempdir_nickname),
                                   num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate,
                                   graph_conv_sizes=graph_conv_sizes, dense_size=dense_size, gpu=gpu)
        else:
            # final training
            train_files = sorted([op.join(tasks_dir, f) for f in os.listdir(tasks_dir)
                                  if op.isfile(op.join(tasks_dir, f))])

            return train_multitask_gc(train_files, model_nickname, smiles_field=smiles_field,
                                      y_field=y_field, id_field=id_field,
                                      tempdir=op.join(RESULTS_DIR, tempdir_nickname), num_epochs=num_epochs,
                                      batch_size=batch_size, learning_rate=learning_rate,
                                      graph_conv_sizes=graph_conv_sizes, dense_size=dense_size, gpu=gpu)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tasks', help='directory where all the training sets are stored', type=str)
    parser.add_argument('-tt', '--test_tasks', help='directory where all the test sets are stored', type=str)
    parser.add_argument('-n', '--name', help='how the model will be name, ex: PCtasks', type=str)
    parser.add_argument('-o', '--output', help='nickname for the temporary directory that will be later removed',
                        default='temp_dir', type=str)
    parser.add_argument('-x', '--cv', help='whether we want to run a cross-validation or not', default=True,
                        type=str2bool)
    parser.add_argument('-r', '--refit', help='whether we want to fit a final model on all the data', default=False,
                        type=str2bool)
    parser.add_argument('-b', '--batch', help='batch size', default=128, type=int)
    parser.add_argument('-e', '--epochs', help='number of epochs', default=40, type=int)
    parser.add_argument('-l', '--learningrate', help='learning rate', default=0.001, type=float)
    parser.add_argument('-s', '--smiles_field', help='name of the column containing the molecules SMILES',
                        default='canonical_smiles', type=str)
    parser.add_argument('-y', '--y_field', help='name of the column containing the target value to predict',
                        default='label', type=str)
    parser.add_argument('-i', '--id_field', help='name of the column containing the identifier for the compounds',
                        default='mol_index', type=str)
    parser.add_argument('-f', '--split_field', help='name of the column containing the fold assignment for CV',
                        default='fold', type=str)
    parser.add_argument('-g', '--gpu', help='id of the GPU to use for training. Default is None, only CPU usage.',
                        default=None, type=int)
    args = parser.parse_args()
    dispatch_job(tasks_dir=args.tasks, model_nickname=args.name, tempdir_nickname=args.output, cv=args.cv,
                 final=args.refit, batch_size=args.batch, num_epochs=args.epochs, learning_rate=args.learningrate,
                 smiles_field=args.smiles_field,  y_field=args.y_field, split_field=args.split_field,
                 test_tasks_dir=args.test_tasks, gpu=args.gpu)
