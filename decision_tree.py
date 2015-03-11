__author__ = 'fabien'


from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np

df = pd.read_csv('data/churn_cleaned.csv')
df.head()