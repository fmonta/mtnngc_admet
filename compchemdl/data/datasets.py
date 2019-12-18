import os.path as op
import pandas as pd
import numpy as np
import tempfile
from functools import reduce

from deepchem.feat import ConvMolFeaturizer
from deepchem.data import CSVLoader

from compchemdl.data import MODEL_DIR
from compchemdl.utils import ensure_dir


def load_training_data(dataset_files, split_field='Fold', smiles_field='Smiles', y_field='Value',
                       id_field='Compound_No', tempdir=op.join(MODEL_DIR, 'datatemp'), cv=True):
    """
    Given a list of datasets in csv format, read them and prepare them for DeepChem (split if needed, etc.)
    :param dataset_files: path to the csv files containing the training data for each task of interest
    :param split_field: column name in the csv giving the fold assignment for CV. Not used if cv=False
    :param smiles_field: column name in the csv giving the SMILES of the compounds
    :param y_field: column name in the csv giving the experimental value to learn
    :param cv: whether we are also splitting the data by split_field
    :return: list of tasks and the list of ConvMol datasets (one dataset per group in split_field)
    """
    ensure_dir(tempdir)

    df_trains = []
    for dataset_file in dataset_files:
        try:
            df_trains.append(pd.read_csv(dataset_file, sep=','))
        except IOError:  # no test split for example
            df = pd.DataFrame(
                {id_field: [], y_field: [], smiles_field: []})  # create an empty df for missing task
            df_trains.append(df)

    n_tasks = len(dataset_files)
    # Rename the y_field column
    df_trains = [df_train.rename(index=str, columns={y_field: y_field + '_%i' % i}) for i, df_train in
                 enumerate(df_trains)]

    # Merge the individual tasks based on Smiles
    if cv:
        df_train = reduce(lambda x, y: pd.merge(x, y, on=[id_field, smiles_field, split_field], how='outer'),
                          df_trains)
    else:
        df_train = reduce(lambda x, y: pd.merge(x, y, on=[id_field, smiles_field], how='outer'), df_trains)
    # Save the merged train csv in a temporary place
    dataset_file = op.join(tempdir, 'data.csv')
    df_train.to_csv(dataset_file, na_rep=np.nan, index=False)

    # Featurization
    featurizer = ConvMolFeaturizer()
    loader = CSVLoader(tasks=[y_field + '_%i' % i for i in range(n_tasks)], smiles_field=smiles_field,
                       featurizer=featurizer, id_field=id_field)
    dataset = loader.featurize(dataset_file, shard_size=8192, data_dir=tempdir)

    if cv:
        folds = np.unique(df_trains[0][split_field].tolist())

        # Separate in folds
        folds_datasets = []
        fold_dirs = []
        for f in folds:
            fold_dir = tempfile.mkdtemp(prefix=tempdir + '/')
            indices = np.flatnonzero(df_train[split_field] == f)
            folds_datasets.append(dataset.select(indices, select_dir=fold_dir))
            fold_dirs.append(fold_dir)

        return ['Value_%i' % i for i in range(n_tasks)], folds_datasets, fold_dirs

    return ['Value_%i' % i for i in range(n_tasks)], dataset


def load_inference_data(dataset_file, n_tasks, tempdir, smiles_field='Smiles', id_field='Compound_No'):
    """
    :param dataset_file: path to the csv files containing the data we want to predict
    :param smiles_field: column name in the csv giving the SMILES of the compounds
    :param id_field: column name in the csv giving the identifier for the molecules
    :param tempdir: directory where ConvMol datasets will be temporarily stored
    :return: list of tasks and the ConvMol datasets
    """
    # Featurize the dataset for Graph Convolutional architecture
    featurizer = ConvMolFeaturizer()
    loader = CSVLoader(tasks=['Value_%i' % i for i in range(n_tasks)], smiles_field=smiles_field, featurizer=featurizer,
                       id_field=id_field)
    dataset = loader.featurize(dataset_file, shard_size=8192, data_dir=tempdir)

    return ['Value_%i' % i for i in range(n_tasks)], dataset


def untransform_y_task(y, task_index, transformer):
    """
    :param y: task values (or predictions) to transform back
    :param task_index: index of the task of interest
    :param transformer: NormalizationTransformer object that was used to transform the data
    :return: back transformed y
    """
    y_std = transformer.y_stds[task_index]
    y_mean = transformer.y_means[task_index]

    return y * y_std + y_mean


def input_smi_to_csv(input_file, outdir, n_tasks):

    # 1. Read input
    molids = []
    smis = []
    molid2smi = {}
    with open(input_file, 'r') as reader:
        for line in reader:
            content = line.strip().split('\t')
            assert (len(content) == 2), 'Input file format does not seem to be correct. Expecting .smi format.'
            molids.append(content[1])
            smis.append(content[0])
            molid2smi[content[1]] = content[0]
    molids = [str(mid) for mid in molids]

    # 2. Write temporary csv file for DeepChem
    if outdir is None:
        outdir = '/tmp'
    ensure_dir(outdir)

    task_columns = ['Value_%i' % i for i in range(n_tasks)]
    input_data = pd.DataFrame.from_dict({'molid': molids, 'smiles': smis})
    for v in task_columns:
        input_data[v] = [0 for _ in range(len(molids))]
    temporary_file = op.join(tempfile.mkdtemp(dir=outdir), 'input.csv')
    input_data.to_csv(temporary_file)
    return molids, smis, temporary_file
