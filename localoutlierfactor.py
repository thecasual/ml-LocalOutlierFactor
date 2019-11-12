import numpy as np
from sklearn.neighbors import LocalOutlierFactor
import pandas as pd

# This is useful when you have a bunch of data and you want to find the anomalies


#Get data
df = pd.read_csv('web.csv')
#df = df[['response', 'hour_of_day']]

# Print out metrics
def getmetrics(df):
    for col in df:
        print("Column : {}".format(col))
        for k,v in df[col].value_counts().items():
            print("Key : {} Value : {}".format(k,v))


# Only numbers use this
#X = df.to_numpy()

# Drop noisy columns
df = df.drop(['timestamp', 'agent', 'url'], axis = 1)

# Convert strings to numbers
X = pd.get_dummies(df, columns=df.columns, drop_first=True).to_numpy()

getmetrics(df)

# n_neighbors default is 20 using this for now
# This is actually running the algorithm so may take a while depending on your hardware
clf = LocalOutlierFactor(n_neighbors=20, n_jobs=-1).fit(X)
Z = clf.negative_outlier_factor_


# Z.min() is the piece of data that is furthest away!

outlier = []
for i in range(len(Z)):
  if Z[i] == Z.min():
      outlier.append(df.values[i])
      #outlier.append(X[i])

# Outlier is now a list of outliers matching the MAXIMUM difference between normal

print(len(outlier))
print(outlier)