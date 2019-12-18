"""
Modified from DeepChem to allow for temporary directory settings
"""

import division
import unicode_literals

import os
import numpy as np


def undo_transforms(y, transformers):
    """Undoes all transformations applied."""
    # Note that transformers have to be undone in reversed order
    for transformer in reversed(transformers):
        if transformer.transform_y:
            y = transformer.untransform(y)
    return y


def undo_grad_transforms(grad, tasks, transformers):
    for transformer in reversed(transformers):
        if transformer.transform_y:
            grad = transformer.untransform_grad(grad, tasks)
    return grad


def get_grad_statistics(dataset):
    """Computes and returns statistics of a dataset

    This function assumes that the first task of a dataset holds the energy for
    an input system, and that the remaining tasks holds the gradient for the
    system.
    """
    if len(dataset) == 0:
        return None, None, None, None
    y = dataset.y
    energy = y[:, 0]
    grad = y[:, 1:]
    for i in range(energy.size):
        grad[i] *= energy[i]
    ydely_means = np.sum(grad, axis=0) / len(energy)
    return grad, ydely_means


class Transformer(object):
    """
    Abstract base class for different ML models.
    """
    # Hack to allow for easy unpickling:
    # http://stefaanlippens.net/pickleproblem
    __module__ = os.path.splitext(os.path.basename(__file__))[0]

    def __init__(self, transform_X=False, transform_y=False, transform_w=False, dataset=None):
        """Initializes transformation based on dataset statistics."""
        self.dataset = dataset
        self.transform_X = transform_X
        self.transform_y = transform_y
        self.transform_w = transform_w
        # One, but not both, transform_X or tranform_y is true
        assert transform_X or transform_y or transform_w
        # Use fact that bools add as ints in python
        assert (transform_X + transform_y + transform_w) == 1

    def transform_array(self, X, y, w):
        """Transform the data in a set of (X, y, w) arrays."""
        raise NotImplementedError("Each Transformer is responsible for its own transform_array method.")

    def untransform(self, z):
        """Reverses stored transformation on provided data."""
        raise NotImplementedError("Each Transformer is responsible for its own untransfomr method.")

    def transform(self, dataset, outdir, parallel=False):
        """
        Transforms all internally stored data.
        Adds X-transform, y-transform columns to metadata.
        NEW: outdir allows to save the transformed DiskDataset somewhere else than in /tmp
        """
        _, y_shape, w_shape, _ = dataset.get_shape()
        if y_shape == tuple() and self.transform_y:
            raise ValueError("Cannot transform y when y_values are not present")
        if w_shape == tuple() and self.transform_w:
            raise ValueError("Cannot transform w when w_values are not present")
        return dataset.transform(lambda X, y, w: self.transform_array(X, y, w), out_dir=outdir)

    def transform_on_array(self, X, y, w):
        """
        Transforms numpy arrays X, y, and w
        """
        X, y, w = self.transform_array(X, y, w)
        return X, y, w


class NormalizationTransformer(Transformer):

    def __init__(self, transform_X=False, transform_y=False, transform_w=False, dataset=None,
                 transform_gradients=False, move_mean=True):
        """Initialize normalization transformation."""
        if transform_X:
            X_means, X_stds = dataset.get_statistics(X_stats=True, y_stats=False)
            self.X_means = X_means
            self.X_stds = X_stds
        elif transform_y:
            y_means, y_stds = dataset.get_statistics(X_stats=False, y_stats=True)
            self.y_means = y_means
            # Control for pathological case with no variance.
            y_stds = np.array(y_stds)
            y_stds[y_stds == 0] = 1.
            self.y_stds = y_stds
        self.transform_gradients = transform_gradients
        self.move_mean = move_mean
        if self.transform_gradients:
            true_grad, ydely_means = get_grad_statistics(dataset)
            self.grad = np.reshape(true_grad, (true_grad.shape[0], -1, 3))
            self.ydely_means = ydely_means

        super(NormalizationTransformer, self).__init__(transform_X=transform_X,
                                                       transform_y=transform_y,
                                                       transform_w=transform_w,
                                                       dataset=dataset)

    def transform(self, dataset, outdir, parallel=False):
        return super(NormalizationTransformer, self).transform(dataset, parallel=parallel, outdir=outdir)

    def transform_array(self, X, y, w):
        """Transform the data in a set of (X, y, w) arrays."""
        if self.transform_X:
            if not hasattr(self, 'move_mean') or self.move_mean:
                X = np.nan_to_num((X - self.X_means) / self.X_stds)
            else:
                X = np.nan_to_num(X / self.X_stds)
        if self.transform_y:
            if not hasattr(self, 'move_mean') or self.move_mean:
                y = np.nan_to_num((y - self.y_means) / self.y_stds)
            else:
                y = np.nan_to_num(y / self.y_stds)
        return (X, y, w)

    def untransform(self, z):
        """
        Undo transformation on provided data.
        """
        if self.transform_X:
            if not hasattr(self, 'move_mean') or self.move_mean:
                return z * self.X_stds + self.X_means
            else:
                return z * self.X_stds
        elif self.transform_y:
            y_stds = self.y_stds
            y_means = self.y_means
            n_tasks = self.y_stds.shape[0]
            z_shape = list(z.shape)
            # Get the reversed shape of z: (..., n_tasks, batch_size)
            z_shape.reverse()
            # Find the task dimension of z
            for dim in z_shape:
                if dim != n_tasks and dim == 1:
                    # Prevent broadcasting on wrong dimension
                    y_stds = np.expand_dims(y_stds, -1)
                    y_means = np.expand_dims(y_means, -1)
            if not hasattr(self, 'move_mean') or self.move_mean:
                return z * y_stds + y_means
            else:
                return z * y_stds

    def untransform_grad(self, grad, tasks):
        """
        Undo transformation on gradient.
        """
        if self.transform_y:

            grad_means = self.y_means[1:]
            energy_var = self.y_stds[0]
            grad_var = 1 / energy_var * (self.ydely_means - self.y_means[0] * self.y_means[1:])
            energy = tasks[:, 0]
            transformed_grad = []

            for i in range(energy.size):
                Etf = energy[i]
                grad_Etf = grad[i].flatten()
                grad_E = Etf * grad_var + energy_var * grad_Etf + grad_means
                grad_E = np.reshape(grad_E, (-1, 3))
                transformed_grad.append(grad_E)

            transformed_grad = np.asarray(transformed_grad)
            return transformed_grad
