import numpy as np
from sklearn.neighbors import LocalOutlierFactor
import pandas as pd
import matplotlib.pyplot as plt

# This is useful when you have a bunch of data and you want to find the anomalies

#Get data
#og = pd.read_csv('web.csv')
og = pd.read_csv('web2.csv')
df = og[['hour_of_day', 'geo.src']]
#df = df[['response', 'hour_of_day']]


def getmetrics(df):
    data = []
    for col in df:
        rank = 1
        for k,v in df[col].value_counts().items():
            data.append([col,k,v,rank])
            rank +=1
    metrics = pd.DataFrame(data, columns = ['Column', 'Object', 'Count', 'Rank'])
    metrics.to_csv('metrics.csv')
    return metrics

metrics = getmetrics(df)

# Convert string characters to proper characters
X = pd.get_dummies(df, columns=df.columns, drop_first=True)
X.to_csv('dummy.csv')

X = X.to_numpy()
clf = LocalOutlierFactor(n_neighbors=35, n_jobs=-1, contamination='auto' ).fit(X)

# Pretty sure this one is wrong or used for something else
#Z = clf.negative_outlier_factor_
Z = clf._lrd

outlier = list((Z).argsort()[:10])
anomaliestocsv = []
for i in outlier:
  anomaliestocsv.append(list(og.iloc[i]))

print(anomaliestocsv)
dataout = pd.DataFrame(anomaliestocsv, columns=og.columns)
dataout.to_csv('anomalies.csv', index=False)