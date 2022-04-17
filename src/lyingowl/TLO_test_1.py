from superlearner import SuperLearner
import pandas as pd
import time
import os

startTime = time.time()


df = pd.read_excel('IMDB Banana Ascii.xlsx')
sentences = df[0].to_numpy()
labels = df[1].to_numpy()

sl = SuperLearner(categorized_data=sentences[:500],
                  categorized_labels=labels[:500],
                  prediction_data=sentences[200:205],
                  testing=False,
                  directory='/Users/nizarmichaud/Desktop/superlearner/',
                  hyperparameters_optimizer='sklearn')
                  
sl.fit()
sl.save_data()
sl.heatmaps()

print(f'Process took: {time.time() - startTime}')
