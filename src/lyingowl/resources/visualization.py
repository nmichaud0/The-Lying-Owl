import pandas as pd
import plotly.graph_objects as go


class AccHeatmap(go.Figure):
    def __init__(self, df, title, x_title, y_title, z_range, **kwargs):
        super().__init__(**kwargs)
        self._df = df
        self._title = title
        self._x_title = x_title
        self._y_title = y_title
        self._z_range = z_range
        self._heatmap = None

    def plot(self):
        fig = go.Figure(data=go.Heatmap(
            z=self._df['accuracy'],
            x=self._df['language model'],
            y=self._df['model'],
            colorscale='Bluered',
            zmin=self._z_range[0],
            zmax=self._z_range[1]
        ))
        fig.update_layout(
            title=self._title,
            xaxis_title=self._x_title,
            yaxis_title=self._y_title,
        )

        self._heatmap = fig

    def write_html(self, filename):
        self._heatmap.write_html(filename)


if __name__ == '__main__':

    # Initialize heatmap df from ML output

    from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score

    # Accuracy measure:
    acc_mes = cohen_kappa_score
    acc_mes_label = "Cohen's Kappa"

    df = pd.read_excel('/Users/nizarmichaud/Downloads/superlearner_testing 3/TLO/data/betas_and_metrics.xlsx')
    df = df.drop(columns=['Unnamed: 0'])

    #df_models = df.drop(columns=['true_values'])
    models = df['models_fitted_labels'].to_list()

    def tokenize_models(models):
        return [model.split('_') for model in models]

    models_tokenized = tokenize_models(models)
    structure = {'model': 0, 'language model': 1, 'supervised learner': 2, 'soft_hard': 3}

    x = []  # model
    y = []  # language model + supervised learner
    z = []  # accuracy

    for i, model in enumerate(models_tokenized):

        if model[structure['supervised learner']] == 'True':
            supervised = ' Hyperparameter Tuning'
        else:
            supervised = ' No Hyperparameter Tuning'

        x.append(model[structure['model']] + supervised)
        y.append(model[structure['language model']])
    z = df['test_cohen_kappa_score'].tolist()

    df_acc = pd.DataFrame({'model': x, 'language model': y, 'accuracy': z})
    df_acc = df_acc.sort_values(by=['model', 'language model'])
    print(df_acc)
    if acc_mes == cohen_kappa_score:
        z_range_ = [0, 1]
    else:
        z_range_ = [.5, 1]
    # Initialize heatmap plot
    heatmap = AccHeatmap(df_acc, acc_mes_label, 'Language model', 'Model', z_range=z_range_)
    heatmap.plot()
