import pandas as pd
import numpy as np
import os
from nnls_opt import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
import optuna
import time

path = '/Users/nizarmichaud/Downloads/joe_dutch_clean_sub_super_learner ACA_Overnight.xlsx'
df = pd.read_excel(path)

df = df.drop(columns=['Unnamed: 0'])

true_values = df['true_values'].to_list()
df_models = df.drop(columns=['true_values'])
models = df_models.columns.to_list()


def tokenize_models(models_labels):
    return [model.split('_') for model in models_labels]


models_tokenized = tokenize_models(models)
structure = {'model': 0, 'language model': 1, 'supervised learner': 2, 'soft_hard': 3}

proba_models = [models[i] for i in range(len(models)) if models_tokenized[i][structure['soft_hard']] == 'proba']

validation_df = df_models.iloc[:int(len(df_models) * 0.5)]
validation_true_values = true_values[:int(len(df_models) * 0.5)]
validation_matrix = validation_df.values
test_df = df_models.iloc[int(len(df_models) * 0.5):]
test_true_values = true_values[int(len(df_models) * 0.5):]
test_matrix = test_df.values

betas = combiner_solve(validation_matrix, validation_true_values)
print(betas, '\n\n\n')
print(sum(betas), '\n\n\n')

val_pred = []
test_pred = []
for i, item in enumerate(validation_true_values):
    val_predictions = round(sum(validation_matrix[i, :] * betas))
    val_pred.append(val_predictions)

    test_predictions = round(sum(test_matrix[i, :] * betas))
    test_pred.append(test_predictions)

b_acc_val = balanced_accuracy_score(validation_true_values, val_pred)
kappa_val = cohen_kappa_score(validation_true_values, val_pred)

b_acc_test = balanced_accuracy_score(test_true_values, test_pred)
kappa_test = cohen_kappa_score(test_true_values, test_pred)

print(f"Balanced accuracy validation:{b_acc_val},\n Cohen's kappa validation {kappa_val}\n\n")
print(f"Balanced accuracy test:{b_acc_test},\n Cohen's kappa test {kappa_test}\n\n")

# Trying with optuna
# SuperLearner optimization with optuna
trial_index = 0


def objective(trial):
    global trial_index
    trial_index += 1
    betas_ = []
    for i in range(validation_matrix.shape[1]):
        betas_.append(trial.suggest_float(f'beta_{i}', 0, 10))

    betas_ = np.round(np.array(betas_) * 1000000)
    betas_count = np.sum(betas_)

    pre_pred = []
    for i in range(validation_matrix.shape[0]):
        pi = 0
        for j in range(len(betas_)):
            pi += betas_[j] * validation_matrix[i, j]

        pre_pred.append(pi/betas_count)

    # pre_predictions_ = validation_matrix * betas_.reshape((betas_.size, 1))

    predictions = np.round(pre_pred)

    bacc = balanced_accuracy_score(validation_true_values, predictions)
    # bacc = cohen_kappa_score(validation_true_values, predictions)
    print(f"Trial: {trial_index} Balanced accuracy: {bacc}")
    # time.sleep(5)
    return bacc


def weighted_predictions(matrix, betas_):
    predictions_ = []
    for i in range(matrix.shape[0]):
        pi = 0
        for j in range(len(betas_)):
            pi += betas_[j] * matrix[i, j]

        predictions_.append(pi/np.sum(betas_))

    return np.round(predictions_)


# TODO: Check here to optimize more the hyperparameters
search_space = {}
for i in range(validation_matrix.shape[1]):
    search_space[f'beta_{i}'] = [0, 10]

optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.RandomSampler())
study.optimize(objective, timeout=60*10)

betas = np.round(np.array(list(study.best_params.values())) * 1000000)
betas_count = np.sum(betas)

# ----------------------------------------------------------------------------------------------------------------------
# Validation
val_pred = weighted_predictions(validation_matrix, betas)

test_pred = weighted_predictions(test_matrix, betas)

b_acc_val = balanced_accuracy_score(validation_true_values, val_pred)
kappa_val = cohen_kappa_score(validation_true_values, val_pred)

b_acc_test = balanced_accuracy_score(test_true_values, test_pred)
kappa_test = cohen_kappa_score(test_true_values, test_pred)

print(f"Balanced accuracy validation:{b_acc_val},\n Cohen's kappa validation {kappa_val}\n\n")
print(f"Balanced accuracy test:{b_acc_test},\n Cohen's kappa test {kappa_test}\n\n")

"""Balanced accuracy validation:0.8289027149321266,
 Cohen's kappa validation 0.6923992855725343


Balanced accuracy test:0.723964063072974,
 Cohen's kappa test 0.48109491925955095"""