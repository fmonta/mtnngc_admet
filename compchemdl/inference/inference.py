import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import os.path as op
import shutil
import pickle
import math
import numpy as np
import pandas as pd
from deepchem.feat import ConvMolFeaturizer
from deepchem.feat.mol_graphs import ConvMol
from deepchem.data import CSVLoader
import tensorflow as tf
import tempfile
import functools
from compchemdl.inference import DATA_DIR
from compchemdl.utils import ensure_dir, ensure_dir_from_file, merge_dicts


# Memory and CPU usage restrictions for tensorflow
config_cpu = tf.ConfigProto(
    device_count={'GPU': 0, 'CPU': 1},
    allow_soft_placement=True,
    inter_op_parallelism_threads=4,
    intra_op_parallelism_threads=4
)


TASKS = ['LOD2', 'LOA2', 'LOS2', 'LMP2']
MODEL = op.join(DATA_DIR, 'PCtasksCS', 'final_model_2018-10-05',
                'bs=128_lr=0.00100_infeat=75_outfeat=128_dense=256_epo=40_rs=123', 'model.ckpt')

####################
# HELPER FUNCTIONS
####################


def transform_y_task(y, task_index, transformer):
    """
    :param y: task array to transform
    :param task_index: index of task on which to apply the transformer
    :param transformer: z-scaler from DeepChem
    :return: z-scaled array for the task of interest
    """
    y_std = transformer.y_stds[task_index]
    y_mean = transformer.y_means[task_index]
    return np.nan_to_num((y - y_mean) / y_std)


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


######################
# DATA PREPARATION
######################


def load_data(dataset_files, tempdir, smiles_field='Smiles', y_field='Value', id_field='Compound_No'):
    """
    :param dataset_files: list of paths to the csv files containing the data for each task of interest
    :param smiles_field: column name in the csv giving the SMILES of the compounds
    :param y_field: column name in the csv giving the experimental value to learn. At inference time, this information
    is ignored but has to be provided.
    :param id_field: column name in the csv giving the identifier for the molecules
    :param tempdir: directory where ConvMol datasets will be temporarily stored
    :return: list of tasks and the list of ConvMol datasets
    """
    ensure_dir(tempdir)
    data_dfs = [pd.read_csv(dataset_file, sep=',') for dataset_file in dataset_files]
    n_tasks = len(dataset_files)
    # Rename the y_field column
    data_dfs = [data_df.rename(index=str, columns={'Value': y_field + '_%i' % i}) for i, data_df in enumerate(data_dfs)]
    # Merge the individual tasks based on Smiles
    df_data = functools.reduce(lambda x, y: pd.merge(x, y, on=[id_field, smiles_field], how='outer'), data_dfs)

    # hacky thing to avoid problems with files where only 1 structure is present (problem in DeepChem)
    df_to_add = pd.DataFrame.from_dict({id_field: ['fake'], smiles_field: ['C']})
    df_data = df_data.append(df_to_add)

    # Save the data csv in a temporary place
    dataset_file = op.join(tempdir, 'data.csv')
    df_data.to_csv(dataset_file, na_rep=np.nan, index=False)

    # Featurize the dataset for Graph Convolutional architecture
    featurizer = ConvMolFeaturizer()
    loader = CSVLoader(tasks=['Value_%i' % i for i in range(n_tasks)], smiles_field=smiles_field, featurizer=featurizer,
                       id_field=id_field)
    dataset = loader.featurize(dataset_file, shard_size=8192, data_dir=tempdir)

    return ['Value_%i' % i for i in range(n_tasks)], dataset


#######################
# PREDICTION
#######################

def predict_operation(X, adjacency_ph, atom_feats_ph, degree_slice_ph, membership_ph, predict_ops, session):

    atoms = X.get_atom_features()
    deg_adj_lists = [X.deg_adj_lists[deg] for deg in range(1, 10 + 1)]

    # Generate dicts
    deg_adj_dict = dict(list(zip(adjacency_ph, deg_adj_lists)))
    atoms_dict = {atom_feats_ph: atoms, degree_slice_ph: X.deg_slice, membership_ph: X.membership}
    atom_dict = merge_dicts([atoms_dict, deg_adj_dict])

    # Run the prediction operation(s)
    resultis = []
    for _, oper in enumerate(predict_ops):
        results_i = list(session.run(oper, feed_dict=atom_dict))
        resultis.append(results_i)
    results = np.vstack(resultis).T  # shape batch_size, n_tasks

    return results


def predict(dataset, checkpoint, task_indices, tempdir, smiles_field='Canonical_Smiles', y_field='Value',
            id_field='Compound_No', batch_size=128):
    """

    :param dataset: path to the data to predict (in csv format with smiles, y (or fake y), molid)
    :param task_indices: list of indices of tasks for which we want a predictions
    :param checkpoint: path to the checkpoint of the trained model
    :param tempdir: where the DiskDataset object will be created for the dataset to predict. Folder will be removed
    afterwards.
    :param smiles_field: header for the SMILES column in the dataset
    :param y_field: header for the value to predict column in the dataset
    :param id_field: header for the molid column in the dataset
    :param batch_size: tells the inferor to cut the dataset in batch_size chunks
    :return:
    """
    device = "/cpu:0"
    # 0. Avoid stupid failure
    ensure_dir(tempdir)
    # 1. Get the data into some kind of DeepChem dataset (DiskDataset)
    _, dset = load_data([dataset], tempdir, smiles_field, y_field, id_field)
    molids_processed = dset.ids

    model_dir = op.split(checkpoint)[0]
    print(model_dir)

    with tf.device(device):
        with tf.Session(config=config_cpu) as sess:
            saver1 = tf.train.import_meta_graph(checkpoint + '.meta')
            saver1.restore(sess, tf.train.latest_checkpoint(model_dir))
            graph = tf.get_default_graph()

            # operation we will want to run:
            predict_ops = []
            for ti in task_indices:
                if ti == 0:
                    predict_ops.append(graph.get_tensor_by_name('Squeeze:0'))
                else:
                    predict_ops.append(graph.get_tensor_by_name('Squeeze_%i:0' % ti))

            # Pick the necessary input placeholders from the graph
            tf_atom_feats = graph.get_tensor_by_name('topology_atom_features:0')
            tf_top_da1 = graph.get_tensor_by_name('topology_deg_adj1:0')
            tf_top_da2 = graph.get_tensor_by_name('topology_deg_adj2:0')
            tf_top_da3 = graph.get_tensor_by_name('topology_deg_adj3:0')
            tf_top_da4 = graph.get_tensor_by_name('topology_deg_adj4:0')
            tf_top_da5 = graph.get_tensor_by_name('topology_deg_adj5:0')
            tf_top_da6 = graph.get_tensor_by_name('topology_deg_adj6:0')
            tf_top_da7 = graph.get_tensor_by_name('topology_deg_adj7:0')
            tf_top_da8 = graph.get_tensor_by_name('topology_deg_adj8:0')
            tf_top_da9 = graph.get_tensor_by_name('topology_deg_adj9:0')
            tf_top_da10 = graph.get_tensor_by_name('topology_deg_adj10:0')
            adjacency_placeholders = [tf_top_da1, tf_top_da2, tf_top_da3, tf_top_da4, tf_top_da5, tf_top_da6,
                                      tf_top_da7, tf_top_da8, tf_top_da9, tf_top_da10]
            tf_top_slice = graph.get_tensor_by_name('topology_deg_slice:0')
            tf_top_memb = graph.get_tensor_by_name('topology_membership:0')

            X_b = dset.X
            n_samples = len(X_b)

            # If X_b has more compounds than batch_size, we need to cut it by batches of batch_size
            if n_samples > batch_size:

                n_batches = int(math.ceil(float(n_samples) / batch_size))
                print(n_batches)
                preds = []
                for i in range(n_batches - 1):
                    X = ConvMol.agglomerate_mols(X_b[i*batch_size:i*batch_size+batch_size])
                    results = predict_operation(X, adjacency_placeholders, tf_atom_feats, tf_top_slice, tf_top_memb,
                                                predict_ops, sess)
                    preds.append(results)

                # last batch
                print('Last batch!')
                X = ConvMol.agglomerate_mols(X_b[(n_batches-1)*batch_size:])
                results = predict_operation(X, adjacency_placeholders, tf_atom_feats, tf_top_slice, tf_top_memb,
                                            predict_ops, sess)
                preds.append(results)

                # Put together all the preds for all the batches (currently list of np arrays)
                preds = np.vstack(preds)
                preds = preds[:n_samples, :]
                print(preds.shape)

            else:  # no need to care about batching
                X = ConvMol.agglomerate_mols(X_b)
                results = predict_operation(X, adjacency_placeholders, tf_atom_feats, tf_top_slice, tf_top_memb,
                                            predict_ops, sess)
                preds = results[:n_samples, :]
    shutil.rmtree(tempdir)
    return preds, molids_processed


def post_process_predictions(preds, checkpoint, task_indices):
    """
    Undo the transformation (z-scaling) of the predictions
    :param preds: numpy array (n_test_cpds, n_tasks) of predictions obtained from calling the predict() method
    :param checkpoint: path to the model snapshot that was used for predictions
    :param task_indices: which of all tasks that MTNN model can predict we are interested in
    :return: un-transformed predictions
    """
    # 1. Find the transformer
    pickled_transformer = op.join(op.dirname(checkpoint), 'transformer.pkl')
    with open(pickled_transformer, 'rb') as reader:
        transformer = pickle.load(reader, encoding='latin1')

    # 2. Apply the untransformation
    untransformed_y = []
    for i, ti in enumerate(task_indices):
        yhat = preds[:, i]
        yhat_new = untransform_y_task(yhat, ti, transformer)
        untransformed_y.append(yhat_new)

    return np.array(untransformed_y).T


#################
# INFERENCE JOB
#################

def run_the_thing(input_file, tasks, output_file=None, tempdir=None):
    """
    :param input_file: file in .smi format (smiles, tab separation, molecule_id)
    :param tasks: lists of tasks among the possible tasks for which we want a prediction
    :param output_file: where to store the predictions (csv format). If None, no output file is written
    :param tempdir: where the temporary directories created by DeepChem will be stored
    :return: predictions (back transformed)
    """

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
    if tempdir is None:
        tempdir = '/tmp'
    ensure_dir(tempdir)
    input_data = pd.DataFrame.from_dict({'molid': molids, 'smiles': smis, 'Value': [0 for _ in range(len(molids))]})
    temporary_file = op.join(tempfile.mkdtemp(dir=tempdir), 'input.csv')
    input_data.to_csv(temporary_file)

    # 3. Sanity check of the requested tasks + convert tasks to ids
    if tasks is None:
        tasks = TASKS  # if no requested tasks, all will be computed
    assert (t in TASKS for t in tasks), 'Task requested is not modeled yet.'
    task_indices = [TASKS.index(t) for t in tasks]

    # 4. Write checkpoint file to have the proper path
    print('writing the checkpoint file paths')
    model_path = op.split(MODEL)[0]
    with open(op.join(model_path, 'checkpoint'), 'w') as writer:
        writer.write('model_checkpoint_path: "')
        writer.write(MODEL)
        writer.write('"')
        writer.write('\n')
        writer.write('all_model_checkpoint_paths: "')
        writer.write(MODEL)
        writer.write('"')

    # 5. Run the prediction
    print('Running the prediction')
    second_tempdir = op.join(tempfile.mkdtemp(dir=tempdir), 'todel')
    ypred, molids_processed = predict(temporary_file, MODEL, task_indices, second_tempdir, smiles_field='smiles',
                                      y_field='Value', id_field='molid')
    # hacky thing to avoid problems with files where only 1 structure is present. Remove the last, fake molecule from
    # the output
    ypred = ypred[:-1, :]
    molids_processed = molids_processed[:-1]
    molids_processed = [str(mid) for mid in molids_processed]

    # 6. Post-process the predictions (remove the z-scaling)
    ypred = post_process_predictions(ypred, MODEL, task_indices)

    # 7. Write down the csv file with the predictions
    if output_file is not None:
        ensure_dir_from_file(output_file)
        print('Writing the predictions on file')
        with open(output_file, 'w') as writer:
            # header
            writer.write(','.join(['CompoundID', 'Canonical Smiles'] + tasks))
            writer.write('\n')
            # content
            if len(molids) == ypred.shape[0]:  # all compounds could be processed
                for mid, smi, preds in zip(molids, smis, ypred):
                    writer.write(','.join([str(mid), smi]))
                    writer.write(',')
                    writer.write(','.join([str(p) for p in preds]))
                    writer.write('\n')
            else:  # then we have to use the list of molids that were processed
                print('Not all compounds could be predicted. Problematic CompoundIDs:')
                print(set(molids).difference(set(molids_processed)))
                for mid, preds in zip(molids_processed, ypred):
                    writer.write(','.join([str(mid), molid2smi[str(mid)]]))
                    writer.write(',')
                    writer.write(','.join([str(p) for p in preds]))
                    writer.write('\n')

    # 8. Delete temp files
    print('Deleting temporary files...')
    shutil.rmtree(op.dirname(temporary_file))
    shutil.rmtree(op.dirname(second_tempdir))

    return molids, smis, ypred


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='Path to the input file containing the compounds to predict in '
                                                        'a .smi format: one line per compound, starting with the '
                                                        'smiles then a tab separation then a molecule id. No other '
                                                        'format is currently accepted.', required=True)
    parser.add_argument('-t', '--tasks', help='list of tasks we want to predict', action='append')
    parser.add_argument('-o', '--output', help='where to save predictions')
    parser.add_argument('-j', '--jobdir', help='temporary directory for intermediate files')
    args = parser.parse_args()
    if args.output:
        run_the_thing(tasks=args.tasks, input_file=args.input, output_file=args.output, tempdir=args.jobdir)
    else:
        print(run_the_thing(tasks=args.tasks, input_file=args.input))
