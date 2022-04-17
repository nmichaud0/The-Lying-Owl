import time

startTime = time.time()

import warnings
from sklearn import exceptions as sklearn_exceptions

for warning in [FutureWarning, sklearn_exceptions.ConvergenceWarning]:
    warnings.filterwarnings("ignore", category=warning)

# ML
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
from sklearn.model_selection import RandomizedSearchCV

# Sentence transformer
from sentence_transformers import SentenceTransformer

# Tokenizers
import nltk

nltk.download('punkt', quiet=True)
import spacy
from textblob import TextBlob

# Utils
import optuna
import numpy as np
import operator
import joblib
import pandas as pd
import os


class ThresholdReachedCallback(object):
    """Threshold Reached Callback for Optuna"""

    # Code from https://github.com/optuna/optuna/issues/1001

    def __init__(self, threshold, direction: str = "minimize") -> None:
        self.direction = direction
        self.threshold = threshold
        self._callback_called = False

        if self.direction == "minimize":
            self._operator = operator.lt
            self._score = np.inf
        elif self.direction == "maximize":
            self._operator = operator.gt
            self._score = -np.inf
        else:
            raise ValueError(f"Expected direction value from ['minimize', 'maximize'], but got {self.direction}")

    def __str__(self) -> str:
        if self._callback_called:
            return f"Objective {self.direction}d and stopped after reaching a threshold value of {self.threshold}"

        return super().__str__()

    def __call__(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """Stop the optimization when threshold value is crossed"""

        self._callback_called = True

        if self._operator(study.best_value, self.threshold):
            study.stop()

        return


class XGBSearching:
    def __init__(self, X, y, accuracy_metric, ntrials=100, algorithm: str = 'optuna' or 'sklearn'):
        self.X = X
        self.y = y
        self.ntrials = ntrials
        self.accuracy_metric = accuracy_metric
        self.algorithm = algorithm
        self.model = xgb.XGBClassifier(eval_metric='auc', use_label_encoder=False)
        self.params = {'n_estimators': np.arange(100, 1000, step=100),
                       'booster': ['gbtree', 'gblinear', 'dart'],
                       'learning_rate': np.arange(0.01, 0.1, step=0.01)}

    def objective(self, trial):
        n_estimators = trial.suggest_int('n_estimators', self.params['n_estimators'][0],
                                         self.params['n_estimators'][-1])
        booster = trial.suggest_categorical('booster', self.params['booster'])
        learning_rate = trial.suggest_float('learning_rate', self.params['learning_rate'][0],
                                            self.params['learning_rate'][-1])

        self.model.set_params(n_estimators=n_estimators,
                              booster=booster,
                              learning_rate=learning_rate,
                              eval_metric='auc',
                              use_label_encoder=False)

        self.model.fit(self.X, self.y)

        return self.accuracy_metric(self.y, self.model.predict(self.X))

    def search(self):
        if self.algorithm in ['optuna', 'sklearn']:
            self.accuracy_metric = balanced_accuracy_score
            print('XGB searching only on Optuna #TODO check this')
            direction = "maximize"
            study = optuna.create_study(direction=direction)
            callbacks = ThresholdReachedCallback(threshold=.999, direction=direction)
            study.optimize(self.objective, n_trials=self.ntrials, callbacks=[callbacks])

            return xgb.XGBClassifier(eval_metric='auc',
                                     use_label_encoder=False,
                                     n_estimators=study.best_params['n_estimators'],
                                     booster=study.best_params['booster'],
                                     learning_rate=study.best_params['learning_rate'])
        elif self.algorithm == 'sklearn_____':

            # Use RandomizedSearchCV to find the best parameters.

            # Create the RSCV object and define its parameters
            print('XGB searching begins')
            rscv = RandomizedSearchCV(self.model, self.params, n_iter=self.ntrials, cv=5, scoring=self.accuracy_metric)
            rscv.fit(self.X, self.y)
            print('XGB searching ends')

            return xgb.XGBClassifier(eval_metric='auc',
                                     use_label_encoder=False,
                                     n_estimators=rscv.best_params_['n_estimators'],
                                     booster=rscv.best_params_['booster'],
                                     learning_rate=rscv.best_params_['learning_rate'])
        else:
            raise ValueError(f'Algorithm: {self.algorithm} not supported. You can use optuna or sklearn.')


class KNNSearching:
    def __init__(self, X, y, accuracy_metric, ntrials=100, algorithm: str = 'optuna' or 'sklearn'):
        self.X = X
        self.y = y
        self.ntrials = ntrials
        self.accuracy_metric = accuracy_metric
        self.algorithm = algorithm
        self.model = KNeighborsClassifier()
        self.params = {'n_neighbors': range(1, 10),
                       'weights': ['uniform', 'distance'],
                       'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                       'leaf_size': np.arange(10, 100, step=10),
                       'p': np.arange(1, 10, step=1),
                       'metric': ['minkowski', 'euclidean', 'manhattan', 'chebyshev']}

    def objective(self, trial):
        n_neighbors = trial.suggest_int('n_neighbors', self.params['n_neighbors'][0], self.params['n_neighbors'][-1])
        weights = trial.suggest_categorical('weights', self.params['weights'])
        algorithm = trial.suggest_categorical('algorithm', self.params['algorithm'])
        leaf_size = trial.suggest_int('leaf_size', self.params['leaf_size'][0], self.params['leaf_size'][-1])
        p = trial.suggest_int('p', self.params['p'][0], self.params['p'][-1])
        metric = trial.suggest_categorical('metric', self.params['metric'])

        self.model.set_params(n_neighbors=n_neighbors,
                              weights=weights,
                              algorithm=algorithm,
                              leaf_size=leaf_size,
                              p=p,
                              metric=metric)

        self.model.fit(self.X, self.y)

        return self.accuracy_metric(self.y, self.model.predict(self.X))

    def search(self):
        if self.algorithm == 'optuna':
            direction = "maximize"
            study = optuna.create_study(direction=direction)
            callbacks = ThresholdReachedCallback(threshold=.999, direction=direction)
            study.optimize(self.objective, n_trials=self.ntrials, callbacks=[callbacks])

            return KNeighborsClassifier(n_neighbors=study.best_params['n_neighbors'],
                                        weights=study.best_params['weights'],
                                        algorithm=study.best_params['algorithm'],
                                        leaf_size=study.best_params['leaf_size'],
                                        p=study.best_params['p'],
                                        metric=study.best_params['metric'])
        elif self.algorithm == 'sklearn':

            # Use RandomizedSearchCV to find the best parameters.

            # Create the RSCV object and define its parameters
            print('KNeighbors searching begins')
            rscv = RandomizedSearchCV(self.model, self.params, n_iter=self.ntrials, cv=5, scoring=self.accuracy_metric)
            rscv.fit(self.X, self.y)
            print('Kneighbors searching ends')

            return KNeighborsClassifier(n_neighbors=rscv.best_params_['n_neighbors'],
                                        weights=rscv.best_params_['weights'],
                                        algorithm=rscv.best_params_['algorithm'],
                                        leaf_size=rscv.best_params_['leaf_size'],
                                        p=rscv.best_params_['p'],
                                        metric=rscv.best_params_['metric'])
        else:
            raise ValueError(f'Algorithm: {self.algorithm} not supported. You can use optuna or sklearn.')


class GNBSearching:
    def __init__(self, X, y, accuracy_metric, ntrials=100, algorithm: str = 'optuna' or 'sklearn'):
        self.X = X
        self.y = y
        self.ntrials = ntrials
        self.accuracy_metric = accuracy_metric
        self.algorithm = algorithm
        self.model = GaussianNB()

    def search(self):
        return self.model


class DTSearching:
    def __init__(self, X, y, accuracy_metric, ntrials=100, algorithm: str = 'optuna' or 'sklearn'):
        self.X = X
        self.y = y
        self.ntrials = ntrials
        self.accuracy_metric = accuracy_metric
        self.algorithm = algorithm
        self.model = DecisionTreeClassifier()
        self.params = {'criterion': ['gini', 'entropy'],
                       'splitter': ['best', 'random'],
                       'min_samples_split': np.arange(2, 5),
                       'min_samples_leaf': np.arange(1, 5),
                       'max_features': ['auto', 'sqrt', 'log2']}

    def objective(self, trial):
        criterion = trial.suggest_categorical('criterion', self.params['criterion'])
        splitter = trial.suggest_categorical('splitter', self.params['splitter'])
        min_samples_split = trial.suggest_int('min_samples_split', self.params['min_samples_split'][0],
                                              self.params['min_samples_split'][-1])
        min_samples_leaf = trial.suggest_int('min_samples_leaf', self.params['min_samples_leaf'][0],
                                             self.params['min_samples_leaf'][-1])
        max_features = trial.suggest_categorical('max_features', self.params['max_features'])

        self.model.set_params(criterion=criterion,
                              splitter=splitter,
                              min_samples_split=min_samples_split,
                              min_samples_leaf=min_samples_leaf,
                              max_features=max_features)

        self.model.fit(self.X, self.y)
        return self.accuracy_metric(self.y, self.model.predict(self.X))

    def search(self):
        if self.algorithm == 'optuna':
            direction = "maximize"
            study = optuna.create_study(direction=direction)
            callbacks = ThresholdReachedCallback(threshold=.999, direction=direction)
            study.optimize(self.objective, n_trials=self.ntrials, callbacks=[callbacks])

            return DecisionTreeClassifier(criterion=study.best_params['criterion'],
                                          splitter=study.best_params['splitter'],
                                          min_samples_split=study.best_params['min_samples_split'],
                                          min_samples_leaf=study.best_params['min_samples_leaf'],
                                          max_features=study.best_params['max_features'])
        elif self.algorithm == 'sklearn':

            # Use RandomizedSearchCV to find the best parameters.

            # Create the RSCV object and define its parameters
            print('DT searching begins')
            rscv = RandomizedSearchCV(self.model, self.params, n_iter=self.ntrials, cv=5, scoring=self.accuracy_metric)
            rscv.fit(self.X, self.y)
            print('DT searching ends')
            
            return DecisionTreeClassifier(criterion=rscv.best_params_['criterion'],
                                          splitter=rscv.best_params_['splitter'],
                                          min_samples_split=rscv.best_params_['min_samples_split'],
                                          min_samples_leaf=rscv.best_params_['min_samples_leaf'],
                                          max_features=rscv.best_params_['max_features'])
        else:
            raise ValueError(f'Algorithm: {self.algorithm} not supported. You can use optuna or sklearn.')


class SVCSearching:
    def __init__(self, X, y, accuracy_metric, ntrials=100, algorithm: str = 'optuna' or 'sklearn'):
        self.X = X
        self.y = y
        self.ntrials = ntrials
        self.accuracy_metric = accuracy_metric
        self.algorithm = algorithm
        self.model = SVC(probability=True)
        self.params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                       'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                       'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
                       'degree': np.arange(1, 10)}

    def objective(self, trial):
        c = trial.suggest_float('C', self.params['C'][0], self.params['C'][-1])
        kernel = trial.suggest_categorical('kernel', self.params['kernel'])
        gamma = trial.suggest_float('gamma', self.params['gamma'][0], self.params['gamma'][-1])
        degree = trial.suggest_int('degree', self.params['degree'][0], self.params['degree'][-1])

        self.model.set_params(C=c, kernel=kernel, gamma=gamma, degree=degree, probability=True)
        self.model.fit(self.X, self.y)
        return self.accuracy_metric(self.y, self.model.predict(self.X))

    def search(self):
        if self.algorithm == 'optuna':
            direction = "maximize"
            study = optuna.create_study(direction=direction)
            callbacks = ThresholdReachedCallback(threshold=.999, direction=direction)
            study.optimize(self.objective, n_trials=self.ntrials, callbacks=[callbacks])

            return SVC(C=study.best_params['C'],
                       kernel=study.best_params['kernel'],
                       gamma=study.best_params['gamma'],
                       degree=study.best_params['degree'],
                       probability=True)
        elif self.algorithm == 'sklearn':

            # Use RandomizedSearchCV to find the best parameters.

            # Create the RSCV object and define its parameters
            print('SVC searching begins')
            rscv = RandomizedSearchCV(self.model, self.params, n_iter=self.ntrials, cv=5, scoring=self.accuracy_metric)
            rscv.fit(self.X, self.y)
            print('SVC searching ends')

            return SVC(C=rscv.best_params_['C'], kernel=rscv.best_params_['kernel'], gamma=rscv.best_params_['gamma'],
                       degree=rscv.best_params_['degree'], probability=True)
        else:
            raise ValueError(f'Algorithm: {self.algorithm} not supported. You can use optuna or sklearn.')


class LRSearching:
    def __init__(self, X, y, accuracy_metric, ntrials=100, algorithm: str = 'optuna' or 'sklearn'):
        self.X = X
        self.y = y
        self.ntrials = ntrials
        self.accuracy_metric = accuracy_metric
        self.algorithm = algorithm
        self.model = LogisticRegression()
        self.params = {'solver': ['liblinear', 'saga', 'lbfgs', 'sag', 'newton-cg'],
                       'max_iter': np.arange(100, 1000, step=100)}

    def objective(self, trial):
        solver = trial.suggest_categorical('solver', self.params['solver'])
        max_iter = trial.suggest_int('max_iter', self.params['max_iter'][0], self.params['max_iter'][-1])

        self.model.set_params(solver=solver, max_iter=max_iter)
        self.model.fit(self.X, self.y)
        return self.accuracy_metric(self.y, self.model.predict(self.X))

    def search(self):

        if self.algorithm == 'optuna':
            direction = "maximize"
            study = optuna.create_study(direction=direction)
            callbacks = ThresholdReachedCallback(threshold=.999, direction=direction)
            study.optimize(self.objective, n_trials=self.ntrials, callbacks=[callbacks])

            return LogisticRegression(solver=study.best_params['solver'],
                                      max_iter=study.best_params['max_iter'])
        elif self.algorithm == 'sklearn':

            # Use RandomizedSearchCV to find the best parameters.

            # Create the RSCV object and define its parameters
            print('LR searching begins')
            rscv = RandomizedSearchCV(self.model, self.params, n_iter=self.ntrials, cv=5, scoring=self.accuracy_metric)
            rscv.fit(self.X, self.y)
            print('LR searching ends')

            return LogisticRegression(solver=rscv.best_params_['solver'], max_iter=rscv.best_params_['max_iter'])
        else:
            raise ValueError(f'Algorithm: {self.algorithm} not supported. You can use optuna or sklearn.')


class RFSearching:
    def __init__(self, X, y, accuracy_metric, ntrials=100, algorithm: str = 'optuna' or 'sklearn'):
        self.X = X
        self.y = y
        self.ntrials = ntrials
        self.accuracy_metric = accuracy_metric
        self.algorithm = algorithm
        self.model = RandomForestClassifier()
        self.params = {'n_estimators': np.arange(10, 1000, step=10),
                       'min_samples_split': np.arange(2, 5),
                       'min_samples_leaf': np.arange(1, 5),
                       'max_features': ['auto', 'sqrt', 'log2']}

    def objective(self, trial):
        n_estimators = trial.suggest_int('n_estimators', self.params['n_estimators'][0],
                                         self.params['n_estimators'][-1])
        min_samples_split = trial.suggest_int('min_samples_split', self.params['min_samples_split'][0],
                                              self.params['min_samples_split'][-1])
        min_samples_leaf = trial.suggest_int('min_samples_leaf', self.params['min_samples_leaf'][0],
                                             self.params['min_samples_leaf'][-1])
        max_features = trial.suggest_categorical('max_features', self.params['max_features'])

        self.model.set_params(n_estimators=n_estimators,
                              min_samples_split=min_samples_split,
                              min_samples_leaf=min_samples_leaf,
                              max_features=max_features)

        self.model.fit(self.X, self.y)
        return self.accuracy_metric(self.y, self.model.predict(self.X))

    def search(self):
        if self.algorithm == 'optuna':
            direction = "maximize"
            study = optuna.create_study(direction=direction)
            callbacks = ThresholdReachedCallback(threshold=.999, direction=direction)
            study.optimize(self.objective, n_trials=self.ntrials, callbacks=[callbacks])

            return RandomForestClassifier(n_estimators=study.best_params['n_estimators'],
                                          min_samples_split=study.best_params['min_samples_split'],
                                          min_samples_leaf=study.best_params['min_samples_leaf'],
                                          max_features=study.best_params['max_features'])
        elif self.algorithm == 'sklearn':

            # Use RandomizedSearchCV to find the best parameters.

            # Create the RSCV object and define its parameters
            print('RF searching begins')
            rscv = RandomizedSearchCV(self.model, self.params, n_iter=self.ntrials, cv=5, scoring=self.accuracy_metric)
            rscv.fit(self.X, self.y)
            print('RF searching ends')

            return RandomForestClassifier(n_estimators=rscv.best_params_['n_estimators'],
                                          min_samples_split=rscv.best_params_['min_samples_split'],
                                          min_samples_leaf=rscv.best_params_['min_samples_leaf'],
                                          max_features=rscv.best_params_['max_features'])
        else:
            raise ValueError(f'Algorithm: {self.algorithm} not supported. You can use optuna or sklearn.')


class MLPSearching:
    def __init__(self, X, y, accuracy_metric, ntrials=100, algorithm: str = 'optuna' or 'sklearn'):
        self.X = X
        self.y = y
        self.ntrials = ntrials
        self.accuracy_metric = accuracy_metric
        self.algorithm = algorithm
        self.model = MLPClassifier()
        max_dim = X.shape[1]
        self.params = {'hidden_layer_sizes': [(i, j) for i in np.arange(10, max_dim, step=int(max_dim / 10))
                                              for j in np.arange(1, 4)],
                       'hidden_layers_units': np.arange(1, max_dim * 2),
                       'hidden_layers_amount': np.arange(1, 10),
                       'activation': ['identity', 'logistic', 'tanh', 'relu'],
                       'solver': ['lbfgs', 'sgd', 'adam'],
                       'alpha': np.flip(10.0 ** -np.arange(1, 10)),
                       'learning_rate': ['constant', 'invscaling', 'adaptive'],
                       'max_iter': np.arange(100, 2000, step=100)}

    def objective(self, trial):
        # hidden_layer = trial.suggest_categorical('hidden_layer_sizes', self.params['hidden_layer_sizes'])
        hidden_layers_units = trial.suggest_int('hidden_layers_units', self.params['hidden_layers_units'][0],
                                                self.params['hidden_layers_units'][-1])
        hidden_layers_amount = trial.suggest_int('hidden_layers_amount', self.params['hidden_layers_amount'][0],
                                                 self.params['hidden_layers_amount'][-1])
        activation = trial.suggest_categorical('activation', self.params['activation'])
        solver = trial.suggest_categorical('solver', self.params['solver'])
        alpha = trial.suggest_float('alpha', self.params['alpha'][0], self.params['alpha'][-1])
        learning_rate = trial.suggest_categorical('learning_rate', self.params['learning_rate'])
        max_iter = trial.suggest_int('max_iter', self.params['max_iter'][0], self.params['max_iter'][-1])

        self.model.set_params(hidden_layer_sizes=(hidden_layers_units, hidden_layers_amount),
                              activation=activation,
                              solver=solver,
                              alpha=alpha,
                              learning_rate=learning_rate, max_iter=max_iter)

        self.model.fit(self.X, self.y)

        return self.accuracy_metric(self.y, self.model.predict(self.X))

    def search(self):
        if self.algorithm == 'optuna':
            direction = "maximize"
            study = optuna.create_study(direction=direction)
            callbacks = ThresholdReachedCallback(threshold=.999, direction=direction)
            study.optimize(self.objective, n_trials=self.ntrials, callbacks=[callbacks])

            return MLPClassifier(
                hidden_layer_sizes=(study.best_params['hidden_layers_units'],
                                    study.best_params['hidden_layers_amount']),
                activation=study.best_params['activation'],
                solver=study.best_params['solver'],
                alpha=study.best_params['alpha'],
                learning_rate=study.best_params['learning_rate'],
                max_iter=study.best_params['max_iter'])
        elif self.algorithm == 'sklearn':

            # Use RandomizedSearchCV to find the best parameters.

            # Create the RSCV object and define its parameters
            print('MLP Searching begins')
            params_sklearn = self.params.copy()
            del params_sklearn['hidden_layers_units']
            del params_sklearn['hidden_layers_amount']
            rscv = RandomizedSearchCV(self.model, params_sklearn, n_iter=self.ntrials, cv=5, scoring=self.accuracy_metric)
            rscv.fit(self.X, self.y)  # Issues with layer size searching... not a generated tuple...
            print('MLP Searching ends')

            return MLPClassifier(hidden_layer_sizes=rscv.best_params_['hidden_layer_sizes'],
                                 activation=rscv.best_params_['activation'],
                                 solver=rscv.best_params_['solver'],
                                 alpha=rscv.best_params_['alpha'],
                                 learning_rate=rscv.best_params_['learning_rate'],
                                 max_iter=rscv.best_params_['max_iter'])
        else:
            raise ValueError(f'Algorithm: {self.algorithm} not supported. You can use optuna or sklearn.')


class BasicClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model=None, searching=False, ntrials=100, accuracy_metric='balanced_accuracy'):
        self.model = model
        self.model_type = self.model.__class__.__name__.lower()
        self.searching = searching
        self.fitted = False
        self.ntrials = ntrials
        self.accuracy_metric = accuracy_metric

        if self.accuracy_metric not in ['balanced_accuracy', 'cohen_kappa_score']:
            raise ValueError('accuracy_metric must be either "balanced_accuracy" or "cohen_kappa_score"')
        if self.accuracy_metric == 'cohen_kappa_score':
            self.accuracy_metric = cohen_kappa_score
        #if self.accuracy_metric == 'balanced_accuracy':
        #    self.accuracy_metric = balanced_accuracy_score

    def __fit__(self, X, y, algorithm):

        if self.searching:

            # get label name based on the sklearn model

            if self.model_type == 'mlpclassifier':
                self.model = MLPSearching(X, y, accuracy_metric=self.accuracy_metric, ntrials=self.ntrials,
                                          algorithm=algorithm).search()

            elif self.model_type == 'randomforestclassifier':
                self.model = RFSearching(X, y, accuracy_metric=self.accuracy_metric, ntrials=self.ntrials,
                                         algorithm=algorithm).search()

            elif self.model_type == 'svc':
                self.model = SVCSearching(X, y, accuracy_metric=self.accuracy_metric, ntrials=self.ntrials,
                                          algorithm=algorithm).search()

            elif self.model_type == 'logisticregression':
                self.model = LRSearching(X, y, accuracy_metric=self.accuracy_metric, ntrials=self.ntrials,
                                         algorithm=algorithm).search()

            elif self.model_type == 'xgbclassifier':
                self.model = XGBSearching(X, y, accuracy_metric=self.accuracy_metric, ntrials=self.ntrials,
                                          algorithm=algorithm).search()

            elif self.model_type == 'decisiontreeclassifier':
                self.model = DTSearching(X, y, accuracy_metric=self.accuracy_metric, ntrials=self.ntrials,
                                         algorithm=algorithm).search()

            elif self.model_type == 'gaussiannb':
                self.model = GNBSearching(X, y, accuracy_metric=self.accuracy_metric, ntrials=self.ntrials,
                                          algorithm=algorithm).search()

            elif self.model_type == 'kneighborsclassifier':
                self.model = KNNSearching(X, y, accuracy_metric=self.accuracy_metric, ntrials=self.ntrials,
                                          algorithm=algorithm).search()

            else:
                raise NotImplementedError(f'Searching is not implemented for this model type: {self.model_type}')

        self.model.fit(X, y)

        self.fitted = True

    def __predict__(self, X):

        if self.fitted:
            return self.model.predict(X)
        else:
            raise RuntimeError('Model has not been fitted yet.')

    def __predict_proba__(self, X):

        if self.fitted:
            return self.model.predict_proba(X)
        else:
            raise RuntimeError('Model has not been fitted yet.')


class Tokenizers:
    def __init__(self, tokenizer_type):
        self.tokenizer_type = tokenizer_type
        self.index = {}
        
        # TODO: Check if this class works good

    def tokenize(self, text):

        Ltext = [i.lower() for i in text]

        if self.tokenizer_type == 'nltk':
            return [nltk.word_tokenize(i) for i in Ltext]
        if self.tokenizer_type == 'spacy':
            nlp = spacy.load(
                'en_core_web_sm')  # Refer to this page if issues encountered:
            # https://github.com/explosion/spaCy/issues/4577
            # Can also try this page: https://anaconda.org/conda-forge/spacy-model-en_core_web_sm
            # Or simply this command with conda: conda install -c conda-forge spacy-model-en_core_web_sm
            return [[token.text for token in nlp(i)] for i in Ltext]
        if self.tokenizer_type == 'textblob':
            return [TextBlob(i).words for i in Ltext]

    def encode(self, text):
        # max sentence size is 1000 words

        sequences = self.tokenize(text)

        # Transform each string of sequences object into a list of integers
        # representing the tokens in the sequence.

        all_words = [i for sublist in sequences for i in sublist]
        all_words = list(set(all_words))
        self.update_index(all_words)

        tokenized_text = [[self.index[i] for i in sublist] for sublist in sequences]
        tokenized_text.append(list(np.zeros(1000)))
        # TODO: find a way to shrink the array to 1000 (in case some people wrote essays instead of answers)

        clean_texts = pd.DataFrame(tokenized_text).fillna(0).to_numpy()
        return clean_texts[:-1]

    def update_index(self, words):

        # Create a dictionary mapping words to integers.
        for word in words:
            if word in self.index.keys():
                continue
            self.index[word] = len(self.index) + 1


class SubSuperLearnerClassifier(BasicClassifier):
    def __init__(self, model=None, searching=False, language_model=None, ntrials=100,
                 accuracy_metric='balanced_accuracy') -> None:
        super().__init__(model=model, searching=searching, ntrials=ntrials, accuracy_metric=accuracy_metric)

        self.language_model = language_model
        self.transformer = None
        self.fitted = False
        self.training_data_transformed = None
        self.training_data = None

        sbert_models = ['distiluse-base-multilingual-cased-v1',
                        'distiluse-base-multilingual-cased-v2',
                        'paraphrase-multilingual-MiniLM-L12-v2',
                        'paraphrase-multilingual-mpnet-base-v2',
                        'all-mpnet-base-v2',
                        'multi-qa-mpnet-base-dot-v1',
                        "gtr-t5-large",
                        "multi-qa-mpnet-base-cos-v1"]

        tokenizers_models = ['nltk', 'spacy', 'textblob']

        if self.language_model is None:
            raise ValueError('Language model must be specified.')
        elif self.language_model in sbert_models:
            self.transformer = SentenceTransformer(self.language_model)
        elif self.language_model in tokenizers_models:
            self.transformer = Tokenizers(self.language_model)
        else:
            raise ValueError('Language model not supported.')

    def fit(self, X, y, algorithm):

        self.training_data = X

        self.__fit__(X, y, algorithm=algorithm)
        self.fitted = True

    def predict(self, X):

        if self.fitted:
            return self.__predict__(X)
        else:
            raise RuntimeError('Model has not been fitted yet.')

    def predict_proba(self, X):

        if self.fitted:
            return self.__predict_proba__(X)
        else:
            raise RuntimeError('Model has not been fitted yet.')


class DataTransformer:
    def __init__(self, transformer: str = None):
        self.transformer_label = transformer

        sbert_models = ['distiluse-base-multilingual-cased-v1',
                        'distiluse-base-multilingual-cased-v2',
                        'paraphrase-multilingual-MiniLM-L12-v2',
                        'paraphrase-multilingual-mpnet-base-v2',
                        'all-mpnet-base-v2',
                        'multi-qa-mpnet-base-dot-v1',
                        "gtr-t5-large",
                        "multi-qa-mpnet-base-cos-v1"]

        tokenizers_models = ['nltk', 'spacy', 'textblob']

        if transformer is None:
            raise ValueError('Language model must be specified.')
        elif transformer in sbert_models:
            self.transformer = SentenceTransformer(self.transformer_label)
        elif transformer in tokenizers_models:
            self.transformer = Tokenizers(self.transformer_label)
        else:
            raise ValueError(f'Language model not supported. {transformer}')

    def transform(self, X):
        print(f'{self.transformer_label}: data beign transformed')
        return self.transformer.encode(X)
