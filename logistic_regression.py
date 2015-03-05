__author__ = 'fabien'
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.learning_curve import validation_curve
from sklearn.preprocessing import StandardScaler
from evaluation import deviance_curve
FIG_SIZE = (16,10)

df = pd.read_csv('data/churn_cleaned.csv')
labels = df['churn']
features = df.drop('churn', axis=1)
features.head()


# Processing

BINARY_FEATURES_NAME = ["Int'l Plan", "VMail Plan"]
FEATURES_NAME = features.columns
continuous_feats = features.drop(BINARY_FEATURES_NAME, axis=1)
binary_feats = features[BINARY_FEATURES_NAME]
# Normalization of continuous variables :
continuous_feats.head()
binary_feats.head()
scaler = StandardScaler()
scaled_feats = pd.DataFrame(scaler.fit_transform(continuous_feats, FEATURES_NAME), columns=continuous_feats.columns)
scaled_feats.head()
normalized_feats = pd.concat([scaled_feats.T, binary_feats.T]).T
df.corr()['churn']

# should probably perform a pca on the continuous variables.

# separating data into training and test set. The latter being kept till the end for performance assessment only
X_train, X_test, y_train, y_test = train_test_split(normalized_feats, labels, train_size=2500, random_state=0)

lr = LogisticRegression(penalty='l2')

param_range = np.arange(0.125, 10.125, 0.125)
metric = "f1"
deviance_curve(lr, X_train, y_train, "C", param_range=param_range, metric=metric, n_folds=5, njobs=-1)



