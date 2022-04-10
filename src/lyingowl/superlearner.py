from __future__ import annotations

import numpy as np

from resources.estimators_2 import *
from resources.optimizers import nnls_optimizer, optuna_optimizer, predict_nnls_optuna
from resources.visualization import AccHeatmap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import os


# ALWAYS USE NNLS_OPTIMIZER FOR SUPERLEARNER ! TODO: Need to check on other datasets

# TODO: Get the data out of the superlearner
#  train/test/validation/betas/validation metrics etc.

# TODO:


class SuperLearner:
    def __init__(self, categorized_data, categorized_labels, prediction_data=None):

        self.categorized_data = categorized_data
        self.categorized_labels = categorized_labels
        self.prediction_data = prediction_data
        self.superlearner_predictions = None

        self.text_data = None
        self.label_data = None

        self.train_text_data = None
        self.train_label_data = None

        self.validation_text_data = None
        self.validation_label_data = None

        self.test_text_data = None
        self.test_label_data = None

        self.fitted = False
        self.betas = None

        self.lang_models = ['distiluse-base-multilingual-cased-v1',
                            'distiluse-base-multilingual-cased-v2',
                            'paraphrase-multilingual-MiniLM-L12-v2',
                            'paraphrase-multilingual-mpnet-base-v2',
                            'all-mpnet-base-v2',
                            'multi-qa-mpnet-base-dot-v1',
                            "gtr-t5-large",
                            "multi-qa-mpnet-base-cos-v1",
                            'nltk',
                            'spacy',
                            'textblob']

        self.est_models = [RandomForestClassifier(),
                           LogisticRegression(),
                           MLPClassifier(),
                           SVC(probability=True),
                           DecisionTreeClassifier(),
                           GaussianNB(),
                           KNeighborsClassifier(),
                           xgb.XGBClassifier(eval_metric='auc', use_label_encoder=False)]

        self.est_models_labels = []

        for lang_model in self.lang_models:
            for est_model in self.est_models:
                for supervising in [True, False]:
                    self.est_models_labels.append(f'{lang_model}_{est_model.__class__.__name__}_{supervising}')

        self.model_count = len(self.est_models_labels)

        self.models_fitted = 0
        self.models_fitted_labels = []

        self.transformed_train_data_dict = {}

        self.training_data = {}
        self.validation_data = {}
        self.test_data = {}
        self.predictions = {}

        self.training_bacc = {}
        self.training_kappa = {}

        self.validation_bacc = {}
        self.validation_kappa = {}

        self.test_bacc = {}
        self.test_kappa = {}

        self.betas_optimizer = None
        self.validation_metrics = None
        self.test_metrics = None

        self.heatmap_data = None

    def fit(self, metric: str = 'balanced_accuracy', beta_optimizer: str = 'nnls') -> None:

        if metric not in ['balanced_accuracy', 'cohen_kappa_score']:
            raise ValueError('metric must be either balanced_accuracy or cohen_kappa_score')

        if beta_optimizer not in ['nnls', 'optuna']:
            raise ValueError('beta_optimizer must be either nnls or optuna')

        self.betas_optimizer = beta_optimizer

        self.train_text_data, test_text_data = train_test_split(self.categorized_data, test_size=0.2,
                                                                random_state=42)
        self.train_label_data, test_label_data = train_test_split(self.categorized_labels, test_size=0.2,
                                                                  random_state=42)

        self.validation_text_data, self.test_text_data = train_test_split(test_text_data, test_size=0.5,
                                                                          random_state=42)
        self.validation_label_data, self.test_label_data = train_test_split(test_label_data, test_size=0.5,
                                                                            random_state=42)

        for lang_model in self.lang_models:
            for est_model in self.est_models:
                for supervising in [True, False]:

                    current_model_label = f'{lang_model}_{est_model.__class__.__name__}_{supervising}'

                    if current_model_label not in self.models_fitted_labels:
                        print(f'Model {current_model_label} already fitted, skipping')
                        continue

                    current_model = SubSuperLearnerClassifier(language_model=lang_model,
                                                              model=est_model,
                                                              searching=supervising,
                                                              ntrials=10)

                    # Fitting the model --------------------------------------------------

                    if lang_model in self.transformed_train_data_dict.keys():
                        current_model.fit(self.transformed_train_data_dict[lang_model], self.train_label_data,
                                          transformed_X=True)
                    else:
                        current_model.fit(self.train_text_data, self.train_label_data, transformed_X=False)
                        self.transformed_train_data_dict[lang_model] = current_model.training_data_transformed

                    # ----------------------------------------------------------------------

                    # Predictions ---------------------------------------------------------
                    if self.prediction_data is not None:
                        self.predictions[current_model_label] = current_model.predict_proba(self.prediction_data)

                    # Evaluating the model ------------------------------------------------

                    training_data_predictions = current_model.predict_proba(self.train_text_data)
                    self.training_bacc[current_model_label] = balanced_accuracy_score(training_data_predictions,
                                                                                      self.train_label_data)
                    self.training_kappa[current_model_label] = cohen_kappa_score(training_data_predictions,
                                                                                 self.train_label_data)
                    self.training_data[current_model_label]['predictions'] = training_data_predictions
                    self.training_data[current_model_label]['labels'] = self.train_label_data

                    validation_data_predictions = current_model.predict_proba(self.validation_text_data)
                    self.validation_bacc[current_model_label] = balanced_accuracy_score(validation_data_predictions,
                                                                                        self.validation_label_data)
                    self.validation_kappa[current_model_label] = cohen_kappa_score(validation_data_predictions,
                                                                                   self.validation_label_data)
                    self.validation_data[current_model_label]['predictions'] = validation_data_predictions
                    self.validation_data[current_model_label]['labels'] = self.validation_label_data

                    test_data_predictions = current_model.predict_proba(self.test_text_data)
                    self.test_bacc[current_model_label] = balanced_accuracy_score(test_data_predictions,
                                                                                  self.test_label_data)
                    self.test_kappa[current_model_label] = cohen_kappa_score(test_data_predictions,
                                                                             self.test_label_data)
                    self.test_data[current_model_label]['predictions'] = test_data_predictions
                    self.test_data[current_model_label]['labels'] = self.test_label_data

                    self.models_fitted += 1
                    self.models_fitted_labels.append(f'{lang_model}_{est_model.__class__.__name__}_{supervising}')

                    print(f'{self.models_fitted}/{self.model_count} models fitted')

                    if self.models_fitted == self.model_count:
                        print('All models fitted')
                        self.fitted = True
                        break

        self.optimize_betas(metric=metric)
        self.superlearner_predictions = self.predict(self.prediction_data)

    def optimize_betas(self, ntrials=100, metric: str = 'balanced_accuracy') -> None:

        if metric not in ['balanced_accuracy', 'cohen_kappa_score']:
            raise ValueError('metric must be either balanced_accuracy or cohen_kappa_score')

        if not self.fitted:
            raise ValueError('All SubSuperLearnerClassifier must be fitted before optimizing betas')

        # Need to concatenate all the predictions from all the models

        validation_predictions = np.array([])
        test_predictions = np.array([])
        for trained_model in self.validation_data.keys():
            validation_predictions = np.column_stack((validation_predictions,
                                                      self.validation_data[trained_model]['predictions']))

            test_predictions = np.column_stack((test_predictions,
                                                self.test_data[trained_model]['predictions']))

        if self.betas_optimizer == 'optuna':
            opt_dict = optuna_optimizer(_predictions_=self.validation_data['predictions'],
                                        true_labels=self.validation_data['labels'],
                                        ntrials=ntrials,
                                        accuracy_metric=metric,
                                        test_set=self.test_data['predictions'],
                                        test_set_labels=self.test_data['labels'])
        else:
            opt_dict = nnls_optimizer(predictions=self.validation_data['predictions'],
                                      true_labels=self.validation_data['labels'],
                                      accuracy_metric=metric,
                                      test_set=self.test_data['predictions'],
                                      test_set_labels=self.test_data['labels'])

        self.betas = opt_dict['betas']
        self.validation_metrics = opt_dict['validation_metrics']
        self.test_metrics = opt_dict['test_metrics']
        print('Betas optimized')

    def predict(self, text_data: np.array) -> np.array:

        if not self.fitted:
            raise RuntimeError('SuperLearner not fitted yet')

        if self.betas is None:
            raise RuntimeError('Betas not set yet, please re-run fit() method')

        if not isinstance(text_data, np.ndarray):
            raise TypeError('text_data must be a list of strings')

        # for models in self.directory_path/models do predictions and multiply by weights and save in
        # self.directory_path/results/prediction_label.csv

        # Need to concatenate all the predictions from all the models

        predictions = np.array([])
        for trained_model in self.models_fitted_labels:
            predictions = np.column_stack((predictions, self.predictions[trained_model]))

        return predict_nnls_optuna(predictions, self.betas, self.betas_optimizer)

    def heatmap(self, data_type=None, metric=None, beta=False):

        if data_type not in ['training', 'validation', 'test', None]:
            raise ValueError(
                'data_type must be either training, validation or test or None if betas heatmap is desired')

        if metric not in ['balanced_accuracy', 'cohen_kappa_score', None]:
            raise ValueError('metric must be either balanced_accuracy or cohen_kappa_score or None if betas heatmap '
                             'is desired')

        if beta:
            data = self.heatmap_data['betas']
        else:
            data = self.heatmap_data[f'{data_type}_{metric}']

        def tokenize_models(models):
            return [model_.split('_') for model_ in models]

        models_tokenized = tokenize_models(self.heatmap_data['models_fitted_labels'])
        structure = {'model': 0, 'language model': 1, 'supervised learner': 2, 'soft_hard': 3}

        z_range = [0, 1] if metric == 'cohen_kappa_score' else [.5, 1]

        x = []
        y = []
        for j, model_ in enumerate(models_tokenized):

            if model_[structure['supervised learner']] == 'True':
                supervised = ' Supervised'
            else:
                supervised = ' Unsupervised'

            x.append(model_[structure['model']] + supervised)
            y.append(model_[structure['language model']])

        df_HM = pd.DataFrame({'model': x,
                              'language model': y,
                              'accuracy': data})
        df_HM = df_HM.sort_values(by=['model', 'language model'])

        # TODO: Warning! Could mess with superlearner

        heatmap = AccHeatmap(df=df_HM,
                             title=f'{data_type.capitalize()} data: {metric}',
                             x_title='Language models',
                             y_title='Models',
                             z_range=z_range)

        heatmap.plot()

    def save_data(self, directory_path: str) -> None:

        # Data to save:
        # 1. betas (specified optimizer) + validation_metrics + test_metrics + models_fitted_labels // heatmaps data
        #  last column is the overall model
        # 2. predictions

        # 1: ---------------------------------------------------------------------------------------------------------
        betas_col = self.betas
        train_bacc_col = []
        train_kappa_col = []
        val_bacc_col = []
        val_kappa_col = []
        test_bacc_col = []
        test_kappa_col = []
        models_df = self.models_fitted_labels.copy()
        for trained_model in self.models_fitted_labels:
            train_bacc_col.append(self.training_bacc[trained_model])
            train_kappa_col.append(self.training_kappa[trained_model])
            val_bacc_col.append(self.validation_bacc[trained_model])
            val_kappa_col.append(self.validation_kappa[trained_model])
            test_bacc_col.append(self.test_bacc[trained_model])
            test_kappa_col.append(self.test_kappa[trained_model])

        betas_col.append(np.nan)
        train_bacc_col.append(np.nan)
        train_kappa_col.append(np.nan)
        val_bacc_col.append(self.validation_metrics['balanced_accuracy'])
        val_kappa_col.append(self.validation_metrics['cohen_kappa_score'])
        test_bacc_col.append(self.test_metrics['balanced_accuracy'])
        test_kappa_col.append(self.test_metrics['cohen_kappa_score'])
        models_df.append('SuperLearner')

        df_1 = pd.DataFrame({'models_fitted_labels': models_df,
                             'betas': betas_col,
                             'training_balanced_accuracy': train_bacc_col,
                             'training_cohen_kappa_score': train_kappa_col,
                             'validation_balanced_accuracy': val_bacc_col,
                             'validation_cohen_kappa_score': val_kappa_col,
                             'test_balanced_accuracy': test_bacc_col,
                             'test_cohen_kappa_score': test_kappa_col})

        self.heatmap_data = df_1.copy()

        # 2: ---------------------------------------------------------------------------------------------------------
        # predictions

        df_2 = pd.DataFrame({'prediction_text': self.prediction_data,
                             'SuperLearner_predictions': self.superlearner_predictions})

        # --------------------------------------------------------------------------------------------------------------
        os.mkdir(directory_path)

        df_1.to_excel(f'{directory_path}/betas_and_metrics.xlsx')
        df_2.to_excel(f'{directory_path}/predictions.xlsx')
        print('Data Saved')


sl = SuperLearner(['bruh', 'bruh2'], [1, 0], '/Users/nizarmichaud/Documents/LyingOwl_SuperLearner_Test_0/')

# Computing everything from data that went out from AWS:
"""
path = '/Users/nizarmichaud/Downloads/joe_dutch_clean_sub_super_learner_clean.xlsx'
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
validation_matrix = validation_df.values
test_df = df_models.iloc[int(len(df_models) * 0.5):]
test_matrix = test_df.values

betas = combiner_solve(validation_matrix, true_values)
print(betas, '\n\n\n')

val_pred = []
test_pred = []
for i, item in enumerate(true_values):

    val_predictions = sum(validation_matrix[i, :] * betas)
    val_pred.append(val_predictions)

    test_predictions = sum(test_matrix[i, :] * betas)
    test_pred.append(test_predictions)

b_acc_val = balanced_accuracy_score(true_values, val_pred)
kappa_val = cohen_kappa_score(true_values, val_pred)

b_acc_test = balanced_accuracy_score(true_values, test_pred)
kappa_test = cohen_kappa_score(true_values, test_pred)

print(f"Balanced accuracy validation:{b_acc_val},\n Cohen's kappa validation {kappa_val}\n\n")
print(f"Balanced accuracy test:{b_acc_test},\n Cohen's kappa test {kappa_test}\n\n")

"""