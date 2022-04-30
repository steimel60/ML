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


#--------------- Model Comparisons --------------------#
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import KBinsDiscretizer

"""
The best way to represent data depends on both the data itself and the model you are using.
Below we will show differences of 2 model types: LinearRegression and DecisionTreeRegressor
"""

X,y = mglearn.datasets.make_wave(n_samples=120)
line = np.linspace(-3,3,1000, endpoint=False).reshape(-1,1)

reg = DecisionTreeRegressor(min_samples_leaf=3).fit(X,y)
#plt.plot(line, reg.predict(line), label = "Decision Tree")

reg = LinearRegression().fit(X,y)
#plt.plot(line, reg.predict(line), label="Linear Regression")

#plt.plot(X[:,0],y,'o',c='k')
#plt.ylabel("Regression Output")
#plt.xlabel("Input Feature")
#plt.legend(loc='best')
#plt.show()

#We can make linear data more powerful by creating bins
kb = KBinsDiscretizer(n_bins=10, strategy='uniform') #10 bins evenly spaced
kb.fit(X)
#Transform data points into bins
X_binned = kb.transform(X)

"""
This process uses one-hot encoding, creating a feature for each bin - a boolean representing if the data point is in the bin
"""

line_binned = kb.transform(line)

reg = LinearRegression().fit(X_binned, y)
#plt.plot(line, reg.predict(line_binned), label='Linear Reg Binned')

reg = DecisionTreeRegressor(min_samples_split=3).fit(X_binned, y)
#plt.plot(line, reg.predict(line_binned), label='Dec Tree Binned')
#plt.vlines(kb.bin_edges_[0], -3,3, linewidth=1, alpha=.2)
#plt.legend(loc='best')
#plt.show()

#We can also add polynomial features
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=10, include_bias=False) #Bias adds feature that's constantly 1
poly.fit(X)

X_poly = poly.transform(X)

reg = LinearRegression().fit(X_poly,y)

line_poly = poly.transform(line)
#plt.plot(line, reg.predict(line_poly), label='Polynomial Lin Reg')

#plt.plot(X[:,0],y,'o',c='k')
#plt.ylabel("Regression Output")
#plt.xlabel("Input Feature")
#plt.legend(loc='best')
#plt.show()


#But more complex models like SVM don't can give better models without feature manipulation
from sklearn.svm import SVR

for gamma in [1,10]:
    svr = SVR(gamma=gamma).fit(X,y)
#    plt.plot(line, svr.predict(line), label=f'SVR Gamma = {gamma}')

#plt.plot(X[:,0],y,'o',c='k')
#plt.ylabel("Regression Output")
#plt.xlabel("Input Feature")
#plt.legend(loc='best')
#plt.show()


#------------------ Housing Market Example -------------------------#
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=0)

#Rescale data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

poly = PolynomialFeatures(degree=2).fit(X_test_scaled)
X_train_poly = poly.transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)
#Simple Ridge Model
ridge = Ridge().fit(X_train_scaled, y_train)
print(f"Score with no interaction: {ridge.score(X_test_scaled, y_test)}")
ridge = Ridge().fit(X_train_poly, y_train)
print(f"Score with interactions: {ridge.score(X_test_poly, y_test)}")
#More complex, RF model
rf = RandomForestRegressor(n_estimators=100).fit(X_train_scaled, y_train)
print(f"Score with no interaction: {rf.score(X_test_scaled, y_test)}")
rf = RandomForestRegressor(n_estimators=100).fit(X_train_poly, y_train)
print(f"Score with no interaction: {rf.score(X_test_poly, y_test)}")
