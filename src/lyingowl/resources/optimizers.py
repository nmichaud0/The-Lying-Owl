import pandas as pd
import numpy as np
from nnls_opt import combiner_solve
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
import optuna

# ------------------------------------------------------------------------------
# Trash imports
from sklearn.model_selection import train_test_split


def nnls_optimizer(predictions: np.array, true_labels: np.array,
                   accuracy_metric: str = "balanced_accuracy", test_set: np.array = None,
                   test_set_labels: np.array = None):
    """
    Optimize the NNLS algorithm.

    Parameters
    ----------
    predictions : numpy.array
        Array of predictions.
    true_labels : numpy.array
        Array of true labels.
    accuracy_metric : str, optional
        The metric to optimize for. The default is "balanced_accuracy".
    test_set : numpy.array, optional
        The test set. The default is None.
    test_set_labels : numpy.array, optional
        The test set labels. The default is None.

    Returns
    -------
    float
        The optimized value.
    """
    betas = combiner_solve(predictions, true_labels)

    validations_predictions = []
    test_predictions = []
    for i, _ in enumerate(true_labels):
        validation_pred = np.round(np.sum(predictions[i, :] * betas))
        validations_predictions.append(validation_pred)
        if test_set is not None:
            test_pred = np.round(np.sum(test_set[i, :] * betas))
            test_predictions.append(test_pred)

    if accuracy_metric == "balanced_accuracy":
        acc_metric = balanced_accuracy_score

    if accuracy_metric == "cohen_kappa_score":
        acc_metric = cohen_kappa_score

    else:
        raise ValueError("The accuracy metric is not supported. '[cohen_kappa_score, balanced_accuracy]'")

    if test_set is not None:
        return {'validation_metrics': {'balanced_accuracy': balanced_accuracy_score(validations_predictions, true_labels),
                                       "cohen's kappa": cohen_kappa_score(validations_predictions, true_labels)},
                'test_set_metrics': {'balanced_accuracy': balanced_accuracy_score(test_predictions, test_set_labels),
                                     "cohen's kappa": cohen_kappa_score(test_predictions, test_set_labels)},
                'betas': betas}
    else:
        return {'validation_metrics': {'balanced_accuracy': balanced_accuracy_score(validations_predictions, true_labels),
                                       "cohen's kappa": cohen_kappa_score(validations_predictions, true_labels)},
                'betas': betas}


def weighted_predictions(matrix, betas_):
    predictions_ = []
    for i in range(matrix.shape[0]):
        pi = sum(betas_[j] * matrix[i, j] for j in range(len(betas_)))
        predictions_.append(pi / np.sum(betas_))

    return np.round(predictions_)


def optuna_optimizer(_predictions_: np.array, true_labels: np.array, ntrials: int = 100,
                     accuracy_metric: str = 'balanced_accuracy', test_set: np.array = None,
                     test_set_labels: np.array = None):
    def objective(trial):
        global _predictions_
        betas_ = [trial.suggest_float(f'beta_{i}', 0, 10) for i in range(_predictions_.shape[1])]

        betas_ = np.round(np.array(betas_) * 1000000000)
        betas_count = np.sum(betas_)

        pre_pred = []
        for i in range(_predictions_.shape[0]):
            pi = sum(betas_[j] * _predictions_[i, j] for j in range(len(betas_)))
            pre_pred.append(pi / betas_count)

        _predictions_ = np.round(pre_pred)

        # bacc = cohen_kappa_score(validation_true_values, predictions)
        return balanced_accuracy_score(true_labels, _predictions_)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=ntrials)
    betas = np.round(np.array(list(study.best_params.values())) * 1000000000)
    betas_count = np.sum(betas)

    if accuracy_metric == "balanced_accuracy":
        acc_metric = balanced_accuracy_score
    if accuracy_metric == "cohen_kappa_score":
        acc_metric = cohen_kappa_score
    else:
        raise ValueError("The accuracy metric is not supported. '[cohen_kappa_score, balanced_accuracy]'")

    validations_predictions = weighted_predictions(_predictions_, betas)

    if test_set is not None:
        test_predictions = weighted_predictions(test_set, betas)

        return {'validation_metrics': {'balanced_accuracy': balanced_accuracy_score(validations_predictions, true_labels),
                                       "cohen's kappa": cohen_kappa_score(validations_predictions, true_labels)},
                'test_set_metrics': {'balanced_accuracy': balanced_accuracy_score(test_predictions, test_set_labels),
                                     "cohen's kappa": cohen_kappa_score(test_predictions, test_set_labels)},
                'betas': betas}
    else:
        return {'validation_metrics': {'balanced_accuracy': balanced_accuracy_score(validations_predictions, true_labels),
                                       "cohen's kappa": cohen_kappa_score(validations_predictions, true_labels)},
                'betas': betas}


def predict_nnls_optuna(X, beta, optimizer: str = 'nnls'):

    if optimizer in {'nnls', 'optuna'}:
        return combiner_solve(X, beta) if optimizer == 'nnls' else weighted_predictions(X, beta)

    else:
        raise ValueError("The optimizer is not supported. '[nnls, optuna]'")
