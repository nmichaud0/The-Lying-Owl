from superlearner import SuperLearner
import pandas as pd

#df = pd.read_excel('TLO/src/lyingowl/joe_dutch_clean.xlsx')

#sentences = df['text'].to_numpy()
#proactive = df['proactive'].to_numpy().astype(int)

df = pd.read_excel('TLO/src/lyingowl/IMDB Banana Ascii.xlsx')
print(df.columns)
text = df[0].to_numpy()
labels = df[1].to_numpy()


sl = SuperLearner(text[:1000], labels[:1000], text[1000:2000], testing=False)
sl.fit()
sl.save_data('superlearner_directory')
sl.heatmap(beta=True)
