import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.decomposition import NMF

#CLEANING
reviews = pd.read_csv('/Users/angelabi/Berkeley/SAAS/MAL-recommend/reviews.csv').drop(labels=['uid','text','link','scores'], axis=1).dropna(axis=0, how="any", inplace=False).drop_duplicates(subset=None, keep='first', inplace=False, ignore_index=False)

#TRANSFORM INTO INPUTTABLE MATRIX
matrix = reviews.pivot(index='profile', columns='anime_uid', values='score').fillna(0).to_numpy()

#SKLEARN
model = NMF(init='nndsvd', max_iter=10)
predicted_matrix = model.fit_transform(matrix)

#EXPORT
df = pd.DataFrame(predicted_matrix)
df.to_csv('/Users/angelabi/Berkeley/SAAS/MAL-recommend/sk_test_2.csv')

#TEST ACCURACY

