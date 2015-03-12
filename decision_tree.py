from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import pylab as pl
import seaborn as sns
from sklearn.tree import export_graphviz
from os import system
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import classification_report
from evaluation import deviance_curve, plot_tree

df = pd.read_csv('data/churn_cleaned.csv')
df.head()
features = df.drop('churn', axis=1)
labels = df['churn']



X_train, X_test, y_train, y_test = train_test_split(features, labels, train_size=2500)
clf = DecisionTreeClassifier(max_features='sqrt', max_depth=3)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
param_range = np.arange(1, 21, 1)
deviance_curve(clf, X_train, y_train, 'max_depth', param_range=param_range, metric='f1', n_folds=5, njobs=1)


plot_tree(clf, 'tree_test.dot')