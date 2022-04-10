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

# Sentence transformer
from sentence_transformers import SentenceTransformer

# Tokenizers
import nltk
import spacy
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from textblob import TextBlob

# Utils
import optuna
import numpy as np


class XGBSearching:
    def __init__(self, X, y, ntrials=100):
        self.X = X
        self.y = y
        self.ntrials = ntrials
        self.model = xgb.XGBClassifier(eval_metric='auc', use_label_encoder=False)
        self.params = {'n_estimators': np.arange(100, 1000, step=100),
                       'booster': ['gbtree', 'gblinear', 'dart'],
                       'learning_rate': np.arange(0.01, 0.1, step=0.01),
                       'gamma': np.arange(0, 1, step=0.01)}

    def objective(self, trial):
        n_estimators = trial.suggest_int('n_estimators', self.params['n_estimators'])
        booster = trial.suggest_categorical('booster', self.params['booster'])
        learning_rate = trial.suggest_uniform('learning_rate', self.params['learning_rate'][0])
        gamma = trial.suggest_uniform('gamma', self.params['gamma'])

        self.model.set_params(n_estimators=n_estimators,
                              booster=booster,
                              learning_rate=learning_rate,
                              gamma=gamma,
                              eval_metric='auc',
                              use_label_encoder=False)

        self.model.fit(self.X, self.y)

        return balanced_accuracy_score(self.y, self.model.predict(self.X))

    def search(self):
        study = optuna.create_study()
        study.optimize(self.objective, n_trials=self.ntrials)

        return xgb.XGBoostClassifier(eval_metric='auc',
                                     use_label_encoder=False,
                                     n_estimators=study.best_params['n_estimators'],
                                     booster=study.best_params['booster'],
                                     learning_rate=study.best_params['learning_rate'],
                                     gamma=study.best_params['gamma'])


class KNNSearching:
    def __init__(self, X, y, ntrials=100):
        self.X = X
        self.y = y
        self.ntrials = ntrials
        self.model = KNeighborsClassifier()
        self.params = {'n_neighbors': range(1, 10),
                       'weights': ['uniform', 'distance'],
                       'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                       'leaf_size': np.arange(10, 100, step=10),
                       'p': np.arange(1, 10, step=1),
                       'metric': ['minkowski', 'euclidean', 'manhattan', 'chebyshev', 'mahalanobis']}

    def objective(self, trial):
        n_neighbors = trial.suggest_int('n_neighbors', self.params['n_neighbors'])
        weights = trial.suggest_categorical('weights', self.params['weights'])
        algorithm = trial.suggest_categorical('algorithm', self.params['algorithm'])
        leaf_size = trial.suggest_int('leaf_size', self.params['leaf_size'])
        p = trial.suggest_int('p', self.params['p'])
        metric = trial.suggest_categorical('metric', self.params['metric'])

        self.model.set_params(n_neighbors=n_neighbors,
                              weights=weights,
                              algorithm=algorithm,
                              leaf_size=leaf_size,
                              p=p,
                              metric=metric)

        self.model.fit(self.X, self.y)

        return balanced_accuracy_score(self.y, self.model.predict(self.X))

    def search(self):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=self.ntrials)

        return KNeighborsClassifier(n_neighbors=study.best_params['n_neighbors'],
                                    weights=study.best_params['weights'],
                                    algorithm=study.best_params['algorithm'],
                                    leaf_size=study.best_params['leaf_size'],
                                    p=study.best_params['p'],
                                    metric=study.best_params['metric'])


class GNBSearching:
    def __init__(self, X, y, ntrials=100):
        self.X = X
        self.y = y
        self.ntrials = ntrials
        self.model = GaussianNB()

    def search(self):
        return self.model


class DTSearching:
    def __init__(self, X, y, ntrials=100):
        self.X = X
        self.y = y
        self.ntrials = ntrials
        self.model = DecisionTreeClassifier()
        self.params = {'criterion': ['gini', 'entropy'],
                       'splitter': ['best', 'random'],
                       'min_samples_split': np.arange(2, 5),
                       'min_samples_leaf': np.arange(1, 5),
                       'max_features': ['auto', 'sqrt', 'log2']}

    def objective(self, trial):
        criterion = trial.suggest_categorical('criterion', self.params['criterion'])
        splitter = trial.suggest_categorical('splitter', self.params['splitter'])
        min_samples_split = trial.suggest_int('min_samples_split', self.params['min_samples_split'])
        min_samples_leaf = trial.suggest_int('min_samples_leaf', self.params['min_samples_leaf'])
        max_features = trial.suggest_categorical('max_features', self.params['max_features'])

        self.model.set_params(criterion=criterion,
                              splitter=splitter,
                              min_samples_split=min_samples_split,
                              min_samples_leaf=min_samples_leaf,
                              max_features=max_features)

        self.model.fit(self.X, self.y)
        return balanced_accuracy_score(self.y, self.model.predict(self.X))

    def search(self):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=self.ntrials)

        return DecisionTreeClassifier(criterion=study.best_params['criterion'],
                                      splitter=study.best_params['splitter'],
                                      min_samples_split=study.best_params['min_samples_split'],
                                      min_samples_leaf=study.best_params['min_samples_leaf'],
                                      max_features=study.best_params['max_features'])


class SVCSearching:
    def __init__(self, X, y, ntrials=100):
        self.X = X
        self.y = y
        self.ntrials = ntrials
        self.model = SVC(probability=True)
        self.params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                       'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                       'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
                       'degree': np.arange(1, 10)}

    def objective(self, trial):
        c = trial.suggest_float('C', self.params['C'])
        kernel = trial.suggest_categorical('kernel', self.params['kernel'])
        gamma = trial.suggest_float('gamma', self.params['gamma'])
        degree = trial.suggest_int('degree', self.params['degree'])

        self.model.set_params(C=c, kernel=kernel, gamma=gamma, degree=degree, probability=True)
        self.model.fit(self.X, self.y)
        return balanced_accuracy_score(self.y, self.model.predict(self.X))

    def search(self):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=self.ntrials)

        return SVC(C=study.best_params['C'],
                   kernel=study.best_params['kernel'],
                   gamma=study.best_params['gamma'],
                   degree=study.best_params['degree'],
                   probability=True)


class LRSearching:
    def __init__(self, X, y, ntrials=100):
        self.X = X
        self.y = y
        self.ntrials = ntrials
        self.model = LogisticRegression()
        self.params = {'penalty': ['l1', 'l2'],
                       'dual': [False, True],
                       'solver': ['liblinear', 'saga', 'lbfgs', 'sag', 'newton-cg'],
                       'max_iter': np.arange(100, 1000, step=100)}

    def objective(self, trial):
        penalty = trial.suggest_categorical('penalty', self.params['penalty'])
        dual = trial.suggest_categorical('dual', self.params['dual'])
        solver = trial.suggest_categorical('solver', self.params['solver'])
        max_iter = trial.suggest_int('max_iter', self.params['max_iter'])

        self.model.set_params(penalty=penalty, dual=dual, solver=solver, max_iter=max_iter)
        self.model.fit(self.X, self.y)
        return balanced_accuracy_score(self.y, self.model.predict(self.X))

    def search(self):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=self.ntrials)
        return LogisticRegression(penalty=study.best_params['penalty'],
                                  dual=study.best_params['dual'],
                                  solver=study.best_params['solver'],
                                  max_iter=study.best_params['max_iter'])


class RFSearching:
    def __init__(self, X, y, ntrials=100):
        self.X = X
        self.y = y
        self.ntrials = ntrials
        self.model = RandomForestClassifier()
        self.params = {'n_estimators': np.arange(10, 1000, step=10),
                       'min_samples_split': np.arange(2, 5),
                       'min_samples_leaf': np.arange(1, 5),
                       'max_features': ['auto', 'sqrt', 'log2']}

    def objective(self, trial):
        n_estimators = trial.suggest_int('n_estimators', self.params['n_estimators'])
        min_samples_split = trial.suggest_int('min_samples_split', self.params['min_samples_split'])
        min_samples_leaf = trial.suggest_int('min_samples_leaf', self.params['min_samples_leaf'])
        max_features = trial.suggest_categorical('max_features', self.params['max_features'])

        self.model.set_params(n_estimators=n_estimators,
                              min_samples_split=min_samples_split,
                              min_samples_leaf=min_samples_leaf,
                              max_features=max_features)

        self.model.fit(self.X, self.y)
        return balanced_accuracy_score(self.y, self.model.predict(self.X))

    def search(self):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=self.ntrials)
        return RandomForestClassifier(n_estimators=study.best_params['n_estimators'],
                                      min_samples_split=study.best_params['min_samples_split'],
                                      min_samples_leaf=study.best_params['min_samples_leaf'],
                                      max_features=study.best_params['max_features'])


class MLPSearching:
    def __init__(self, X, y, ntrials=100):
        self.X = X
        self.y = y
        self.ntrials = ntrials
        self.model = MLPClassifier()
        max_dim = X.shape[1]
        self.params = {'hidden_layer_sizes': [(i, j) for i in np.arange(10, max_dim, step=int(max_dim / 10))
                                              for j in np.arange(1, 4)],
                       'activation': ['identity', 'logistic', 'tanh', 'relu'],
                       'solver': ['lbfgs', 'sgd', 'adam'],
                       'alpha': 10 ** -np.arange(1, 10),
                       'learning_rate': ['constant', 'invscaling', 'adaptive'],
                       'max_iter': np.arange(100, 2000, step=100)}

    def objective(self, trial):
        hidden_layer = trial.suggest_categorical('hidden_layer_sizes', self.params['hidden_layer_sizes'])
        activation = trial.suggest_categorical('activation', self.params['activation'])
        solver = trial.suggest_categorical('solver', self.params['solver'])
        alpha = trial.suggest_float('alpha', self.params['alpha'])
        learning_rate = trial.suggest_categorical('learning_rate', self.params['learning_rate'])
        max_iter = trial.suggest_int('max_iter', self.params['max_iter'])

        self.model.set_params(hidden_layer_sizes=hidden_layer, activation=activation, solver=solver, alpha=alpha,
                              learning_rate=learning_rate, max_iter=max_iter)
        self.model.fit(self.X, self.y)
        return balanced_accuracy_score(self.y, self.model.predict(self.X))

    def search(self):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=self.ntrials)
        return MLPClassifier(hidden_layer_sizes=study.best_params['hidden_layer_sizes'],
                             activation=study.best_params['activation'],
                             solver=study.best_params['solver'],
                             alpha=study.best_params['alpha'],
                             learning_rate=study.best_params['learning_rate'],
                             max_iter=study.best_params['max_iter'])


class BasicClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model=None, searching=False):
        self.model = model
        self.model_type = self.model.__class__.__name__.lower()
        self.searching = searching
        self.fitted = False
        self.ntrials = 100

    def __fit__(self, X, y):

        if self.searching:

            # get label name based on the sklearn model

            if self.model_type == 'mlpclassifier':
                self.model = MLPSearching(X, y, ntrials=self.ntrials).search()

            elif self.model_type == 'randomforestclassifier':
                self.model = RFSearching(X, y, ntrials=self.ntrials).search()

            elif self.model_type == 'svc':
                self.model = SVCSearching(X, y, ntrials=self.ntrials).search()

            elif self.model_type == 'logisticregression':
                self.model = LRSearching(X, y, ntrials=self.ntrials).search()

            elif self.model_type == 'xgboostclassifier':
                self.model = XGBSearching(X, y, ntrials=self.ntrials).search()

            elif self.model_type == 'decisiontreeclassifier':
                self.model = DTSearching(X, y, ntrials=self.ntrials).search()

            elif self.model_type == 'gaussiannb':
                self.model = GNBSearching(X, y, ntrials=self.ntrials).search()

            elif self.model_type == 'kneighborsclassifier':
                self.model = KNNSearching(X, y, ntrials=self.ntrials).search()

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

    def tokenize(self, text):

        Ltext = [i.lower() for i in text]

        if self.tokenizer_type == 'nltk':
            return [nltk.word_tokenize(i) for i in Ltext]
        if self.tokenizer_type == 'spacy':
            nlp = spacy.load('en_core_web_sm')
            return [[token.text for token in nlp(i)] for i in Ltext]
        if self.tokenizer_type == 'tensorflow':
            tf_tokenizer = Tokenizer(num_words=None, oov_token='<OOV>')
            tf_tokenizer.fit_on_texts(Ltext)
            return tf_tokenizer.texts_to_sequences(Ltext)
        if self.tokenizer_type == 'textblob':
            return [TextBlob(i).words for i in Ltext]

    def encode(self, text):

        sequences = self.tokenize(text)

        # Transform each string of sequences object into a list of integers
        # representing the tokens in the sequence.

        all_words = [i for sublist in sequences for i in sublist]
        all_words = list(set(all_words))
        self.update_index(all_words)
        return np.asarray([[self.index[i] for i in sublist] for sublist in sequences])

        # Create a dataframe with the sequences and the corresponding labels.

    #  TODO: check if tensorflow arlready does the encoding

    def update_index(self, words):

        # Create a dictionary mapping words to integers.
        for word in words:
            if word in self.index.keys():
                continue
            self.index[word] = len(self.index) + 1


class SubSuperLearnerClassifier(BasicClassifier):
    def __init__(self, model=None, searching=False, language_model=None):
        super().__init__(model=model, searching=searching)

        self.language_model = language_model
        self.transformer = None
        self.fitted = False

        sbert_models = ['distiluse-base-multilingual-cased-v1',
                        'distiluse-base-multilingual-cased-v2',
                        'paraphrase-multilingual-MiniLM-L12-v2',
                        'paraphrase-multilingual-mpnet-base-v2',
                        'all-mpnet-base-v2',
                        'multi-qa-mpnet-base-dot-v1',
                        "gtr-t5-large",
                        "multi-qa-mpnet-base-cos-v1"]  # TODO: add more models

        tokenizers_models = ['nltk', 'spacy', 'textblob', 'tensorflow']  # TODO: add more models

        if self.language_model is None:
            raise ValueError('Language model must be specified.')
        elif self.language_model in sbert_models:
            self.transformer = SentenceTransformer(self.language_model)
        elif self.language_model in tokenizers_models:
            self.transformer = Tokenizers(self.language_model)
        else:
            raise ValueError('Language model not supported.')

    def fit(self, X, y):

        if self.transformer is None:
            raise ValueError('Language model must be specified.')

        transformed_X = self.transformer.encode(X)

        self.__fit__(transformed_X, y)
        self.fitted = True

    def predict(self, X):

        if self.fitted:
            transformed_X = self.transformer.encode(X)
            return self.__predict__(transformed_X)
        else:
            raise RuntimeError('Model has not been fitted yet.')

    def predict_proba(self, X):

        if self.fitted:
            transformed_X = self.transformer.encode(X)
            return self.__predict_proba__(transformed_X)
        else:
            raise RuntimeError('Model has not been fitted yet.')
