__author__ = 'fabien'
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import tests, SelectKBest, VarianceThreshold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import classification_report
from sklearn.grid_search import GridSearchCV
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
churn_df = pd.read_csv('data/churn.csv')
col_names = churn_df.columns.tolist()
churn_df.head()
for name in col_names:
    print(name)
churn_df[churn_df['Churn?'] == "True."]["Churn?"] = 1
churn_df[churn_df['Churn?'] != 1] = 0
churn_df.head()
churn_df['Churn?'] = churn_df['Churn?'] == "True."
churn_df
labels = churn_df["Churn?"]

np.sum(labels == True)
features = churn_df.drop(['State', 'Area Code', 'Phone', 'Churn?'], axis=1)
yes_no_cols = ["Int'l Plan", "VMail Plan"]
features[yes_no_cols] = (features[yes_no_cols] == 'yes')
features_names = features.columns

scaler = StandardScaler()
X = features.as_matrix().astype(np.float)
X = scaler.fit_transform(X)
y = np.where(labels == 'True.', 1, 0)



X.shape
y.shape
y
labels.describe()
svm_clf = LinearSVC(C=0.5, penalty='l1', dual=False)
svm_clf.fit_transform(X, y).shape
svm_clf.fit(X, y)

scores = cross_val_score(svm_clf, X, y, cv=5)
print(scores)
param_grid = {'C': 10. ** np.arange(-3, 4)}
grid_search = GridSearchCV(svm_clf, param_grid=param_grid, cv=5, verbose=3)
grid_search.fit(X, y)

print(grid_search.best_params_)
print(grid_search.best_score_)

for c in grid_search.grid_scores_:
    print(c.mean_validation_score)


plt.figsize(12, 6)
plt.plot([c.mean_validation_score for c in grid_search.grid_scores_], label="validation accuracy")
plt.xticks(np.arange(6), param_grid['C']); plt.xlabel("C"); plt.ylabel("Accuracy");plt.legend(loc='best');
plt.show()

lr_clf = LogisticRegression(C=0.1, penalty='l1', dual=False)
lr_clf.fit(X, y)

scores = cross_val_score(lr_clf, X, y, cv=5)
print(scores)
param_grid = {'C': 10. ** np.arange(-3, 4)}
grid_search = GridSearchCV(lr_clf, param_grid=param_grid, cv=5, verbose=3)
grid_search.fit(X, y)

print(grid_search.best_params_)
print(grid_search.best_score_)

rf_clf = RandomForestClassifier(n_estimators=200, max_features='auto', max_depth=7, n_jobs=-1)
scores = cross_val_score(rf_clf, X, y)
print(scores)

param_grid = {'max_depth': [16, 32, 48, 64, 72, 96, 128, 144, 256]}
grid_search = GridSearchCV(rf_clf, param_grid=param_grid, cv=3, verbose=2)
grid_search.fit(X, y)
print(grid_search.best_params_)
print(grid_search.best_score_)
plt.plot([c.mean_validation_score for c in grid_search.grid_scores_], label="validation accuracy")
plt.xticks(np.arange(8), param_grid['max_depth']); plt.xlabel("depth"); plt.ylabel("Accuracy");plt.legend(loc='best');

from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(max_depth=7,
                                learning_rate=0.1,
                                n_estimators=150,
                                max_features='auto',
                                verbose=1)

scores = cross_val_score(gb, X, y)
print(scores)
train_test_split(X, y, train_size=2400)
gb.fit(X, y)
np.argmax(gb.feature_importances_)
gbedict
features_names[4]
features_names[6]
len(np.arange(150, 800, 50))

