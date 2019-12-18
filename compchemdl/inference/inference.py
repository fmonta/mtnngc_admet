import os.path as op
import shutil
import pickle
import numpy as np
import tempfile

from deepchem.models import GraphConvModel

from compchemdl.data.datasets import load_inference_data, input_smi_to_csv, untransform_y_task
from compchemdl.utils import ensure_dir, ensure_dir_from_file


def easy_predict(dataset, model, n_tasks, tempdir, smiles_field='Canonical_Smiles', id_field='Compound_No'):
    ensure_dir(tempdir)
    # 1. Get the data into the DeepChem DiskDataset object format
    _, dset = load_inference_data(dataset, n_tasks, tempdir, smiles_field, id_field)
    molids_processed = dset.ids

    predictions = model.predict(dset)
    print(predictions.shape)
    print(predictions[:20, :])

    return predictions, molids_processed


def post_process_predictions(preds, checkpoint):
    """
    Undo the transformation (z-scaling) of the predictions
    :param preds: numpy array (n_test_cpds, n_tasks) of predictions obtained from calling the predict() method
    :param checkpoint: path to the model snapshot that was used for predictions
    :return: un-transformed predictions
    """
    # 1. Find the transformer
    pickled_transformer = op.abspath(op.join(checkpoint, '..', 'transformer.pkl'))
    with open(pickled_transformer, 'rb') as reader:
        transformer = pickle.load(reader, encoding='latin1')

    # 2. Apply the untransformation
    untransformed_y = []
    for i in range(preds.shape[1]):
        yhat = preds[:, i]
        yhat_new = untransform_y_task(yhat, i, transformer)
        untransformed_y.append(yhat_new)

    return np.array(untransformed_y).T


#################
# INFERENCE JOB
#################

def run_the_inference(input_file, model_dir, output_file=None, tempdir=None, gpu=None):
    """
    :param input_file: file in .smi format (smiles, tab separation, molecule_id)
    :param checkpoint_file: path to the saved checkpoint of the model we want to use for inference
    :param output_file: where to store the predictions (csv format). If None, no output file is written
    :param tempdir: where the temporary directories created by DeepChem will be stored
    :param gpu: which GPU to use. If None, only CPU will be used
    :return: predictions (back transformed)
    """
    if gpu is None:
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # we will use CPU only for inference

    else:
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '%i' % gpu

    # 1. Load the model
    model = GraphConvModel.load_from_dir(model_dir, restore=True)
    n_tasks = model.n_tasks

    # 2. Prepare input data
    molids, smis, input_dset = input_smi_to_csv(input_file, tempdir, n_tasks)

    # 3. Run the prediction
    print('Running the prediction')
    second_tempdir = op.join(tempfile.mkdtemp(dir=tempdir), 'todel')
    ypred, molids_processed = easy_predict(input_dset, model, n_tasks, second_tempdir, smiles_field='smiles',
                                           id_field='molid')
    molids_processed = [str(mid) for mid in molids_processed]

    # 4. Post-process the predictions (remove the z-scaling)
    ypred = post_process_predictions(ypred, model_dir)
    print(ypred)

    # 5. Write down the csv file with the predictions
    if output_file is not None:
        ensure_dir_from_file(output_file)
        print('Writing the predictions on file')
        with open(output_file, 'w') as writer:
            # header
            writer.write(','.join(['CompoundID', 'Canonical Smiles'] + ['task_%i' % i for i in range(n_tasks)]))
            writer.write('\n')
            # content
            if len(molids_processed) == len(molids):
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

    # 6. Delete temp files
    print('Deleting temporary files...')
    shutil.rmtree(op.dirname(input_dset))
    shutil.rmtree(op.dirname(second_tempdir))

    return molids, smis, ypred


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='Path to the input file containing the compounds to predict in '
                                                        'a .smi format: one line per compound, starting with the '
                                                        'smiles then a tab separation then a molecule id. No other '
                                                        'format is currently accepted.', required=True)
    parser.add_argument('-c', '--checkpoint', type=str, help='Path to the directory where the model is stored',
                        required=True)
    parser.add_argument('-o', '--output', help='where to save predictions')
    parser.add_argument('-j', '--jobdir', help='temporary directory for intermediate files')
    parser.add_argument('-g', '--gpu', help='id of the GPU to use', default=None, type=int)
    args = parser.parse_args()
    if args.output:
        run_the_inference(input_file=args.input, model_dir=args.checkpoint, output_file=args.output,
                          tempdir=args.jobdir, gpu=args.gpu)
    else:
        print(run_the_inference(input_file=args.input, model_dir=args.checkpoint, gpu=args.gpu))
