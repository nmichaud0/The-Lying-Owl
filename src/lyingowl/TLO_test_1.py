from superlearner import SuperLearner
import pandas as pd

df = pd.read_excel('TLO/src/lyingowl/joe_dutch_clean.xlsx')

sentences = df['text'].to_numpy()
proactive = df['proactive'].to_numpy().astype(int)


sl = SuperLearner(sentences[:-50], proactive[:-50], sentences[-50:-1], testing=True,
                  directory='NEED TO CHECK THIS', hyperparameters_optimizer='sklearn')
sl.fit()
sl.save_data('superlearner_directory')
sl.heatmap(beta=True)
