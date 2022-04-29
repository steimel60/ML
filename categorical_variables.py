#standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mglearn
import os


#---------------- One-Hot Encoding -------------------#
"""
Takes a categorial feature and breaks it into separate features represented with a boolean.
Ex) Consider a dataset that has a feature "Employer Type" with the options: Private, Govt, Self.
We would add these 3 new features replacing the "Employer Type" feature.
Our dataset may now look like the example below

name        Private         Govt        Self
Bob         1               0           0
Sally       0               0           1
Paul        1               0           1

Where Bob and Paul both work for the Private Sector and Sally is Self Employed.
"""

adult_path = os.path.join(mglearn.datasets.DATA_PATH, "adult.data")
data = pd.read_csv(adult_path, header=None, index_col=False,
            names=['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                    'marital-status', 'occupation', 'relationship', 'race', 'gender',
                    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'])

#For illustration purposes we only select some columns
data = data[['age', 'workclass', 'education', 'gender', 'hours-per-week', 'occupation', 'income']]

print(data.head())
print(data.gender.value_counts())

#We can simply use Pandas "get_dummies" to encode the data
data_dummies = pd.get_dummies(data)
print(f"Original Features:\n{data.columns}")
print(f"Dummy Features:\n{data_dummies.columns}")

#Remove target data
features = data_dummies.loc[:, 'age':'occupation_ Transport-moving']
X = features.values #Get numpy array of data
y = data_dummies['income_ >50K'].values #Numpy array of target data

#Build model on converted data
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0) #Split data into training and testing sets
logreg = LogisticRegression().fit(X_train, y_train) #Build and fit model
print(f"Test Score: {logreg.score(X_test, y_test)}")

#Applying different Encoding methods to different columns of the same dataset
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

ct = ColumnTransformer(         #This class allows different endocing methods
    [("scaling", StandardScaler(), ['age', 'hours-per-week']),  #Scale normally for linear data
    ("onehot", OneHotEncoder(sparse=False), ['workclass', 'education', 'gender', 'occupation'])  #Separate into boolean columns like above, regardless of data type in column
    ]
)

data_features = data.drop("income", axis=1)
X_train, X_test, y_train, y_test = train_test_split(data_features, data.income, random_state=0)
ct.fit(X_train)
X_train_trans = ct.transform(X_train)
logreg.fit(X_train_trans, y_train)
X_test_trans = ct.transform(X_test)
print(f"Test Score: {logreg.score(X_test_trans, y_test)}")
