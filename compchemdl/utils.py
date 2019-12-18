import os
import os.path as op
from sklearn.metrics.regression import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score
from scipy.stats import spearmanr
import argparse


def ensure_dir_from_file(file_path):
    directory = op.dirname(file_path)
    if not op.exists(directory):
        os.makedirs(directory)


def ensure_dir(dirpath):
    if not op.exists(dirpath):
        os.makedirs(dirpath)


def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z


def merge_dicts(l):
    """
    Convenience function to merge list of dictionaries.
    """
    merged = {}
    for dictio in l:
        merged = merge_two_dicts(merged, dictio)
    return merged


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def evaluate(ytrue, ypred):
    """

    :param ytrue: true value of the dependent variable, numpy array
    :param ypred: predictions for the dependent variable, numpy array
    :return: different evaluation metrics: R squared, mean square error, mean absolute error, and fraction of variance
    explained and Spearman ranking correlation coefficient
    """
    r2 = r2_score(ytrue, ypred)
    mse = mean_squared_error(ytrue, ypred)
    mae = mean_absolute_error(ytrue, ypred)
    variance_explained = explained_variance_score(ytrue, ypred)
    spearman = spearmanr(ytrue, ypred)[0]
    return r2, mse, mae, variance_explained, spearman
