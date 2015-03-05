__author__ = 'fabien'

import pandas as pd
import numpy as np
from copy import deepcopy

churn_df = pd.read_csv('data/churn.csv')
col_names = churn_df.columns.tolist()
print(col_names)
churn_df['Churn?'] = churn_df['Churn?'] == "True."
labels = churn_df["Churn?"]
features = churn_df.drop(['State', 'Area Code', 'Phone', 'Churn?'], axis=1)
yes_no_cols = ["Int'l Plan", "VMail Plan"]
features[yes_no_cols] = (features[yes_no_cols] == 'yes')
features_names = features.columns
dataframe = deepcopy(features)
dataframe['churn'] = labels
dataframe.to_csv("data/churn_cleaned.csv", index=False)

