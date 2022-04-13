from superlearner import SuperLearner
import pandas as pd
import time

startTime = time.time()

#df = pd.read_excel('TLO/src/lyingowl/joe_dutch_clean.xlsx')

#sentences = df['text'].to_numpy()
#proactive = df['proactive'].to_numpy().astype(int)

df = pd.read_excel('TLO/src/lyingowl/IMDB Banana Ascii.xlsx')
print(df.columns)
sentences = df[0].to_numpy()
labels = df[1].to_numpy()

sl = SuperLearner(categorized_data=sentences[:200],
                  categorized_labels=labels[:200],
                  prediction_data=sentences[200:205],
                  testing=False,
                  directory='superlearner_testing',
                  hyperparameters_optimizer='sklearn')
                  
sl.fit()
sl.save_data()
sl.heatmap(beta=True)

print(f'Process took: {time.time() - startTime}')
