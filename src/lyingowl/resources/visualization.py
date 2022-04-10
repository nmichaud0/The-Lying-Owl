import pandas as pd
import plotly.graph_objects as go


class AccHeatmap:
    def __init__(self, df, title, x_title, y_title, z_range):
        self.df = df
        self.title = title
        self.x_title = x_title
        self.y_title = y_title
        self.z_range = z_range

    def plot(self):
        fig = go.Figure(data=go.Heatmap(
            z=self.df['accuracy'],
            x=self.df['language model'],
            y=self.df['model'],
            colorscale='Bluered',
            zmin=self.z_range[0],
            zmax=self.z_range[1]
        ))
        fig.update_layout(
            title=self.title,
            xaxis_title=self.x_title,
            yaxis_title=self.y_title,
        )
        fig.show()


if __name__ == '__main__':

    # Initialize heatmap df from ML output

    from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score

    # Accuracy measure:
    acc_mes = cohen_kappa_score
    acc_mes_label = "Cohen's Kappa"

    df = pd.read_excel('/Users/nizarmichaud/Downloads/joe_dutch_clean_sub_super_learner ACA_Overnight.xlsx')
    df = df.drop(columns=['Unnamed: 0'])

    true_values = df['true_values'].to_list()
    df_models = df.drop(columns=['true_values'])
    models = df_models.columns

    def tokenize_models(models):
        return [model.split('_') for model in models]

    models_tokenized = tokenize_models(models)
    structure = {'model': 0, 'language model': 1, 'supervised learner': 2, 'soft_hard': 3}

    x = []  # model
    y = []  # language model + supervised learner
    z = []  # accuracy

    for i, model in enumerate(models_tokenized):
        if model[structure['soft_hard']] == 'proba':
            continue

        if model[structure['supervised learner']] == 'True':
            supervised = ' Supervised'
        else:
            supervised = ' Unsupervised'

        x.append(model[structure['model']] + supervised)
        y.append(model[structure['language model']])
        print(model, df_models[models[i]].to_list()[0])
        z.append(acc_mes(true_values, df_models[models[i]].to_list()))

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
