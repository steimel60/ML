#standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data_url = "https://www.kaggle.com/datasets/vardhansiramdasu/fraudulent-transactions-prediction?resource=download"


#------------------ import data ----------------------#
df = pd.read_csv('C:/Users/Dylan/Documents/Python/ML/Projects/FraudTransactions/Fraud.csv')

#print(df.head())          #view df
for col in df.columns:
    print(col)             #Print cols


arr = np.array(df['type']) #Transactions types as np array
arr = np.unique(arr)       #Get classes of transactions
#print(arr)                #print them

#Convert strings to ints
for i in range(len(arr)):
    df['type'].replace(to_replace=arr[i], value=i, inplace=True)

#print(df.head())          #view df


#----------- decision_function -------------------#
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split


#List of important features
features = ['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']

#Split into test and train data
X_train, X_test, y_train, y_test = train_test_split(df[features], df['isFraud'])

gbrt = GradientBoostingClassifier().fit(X_train, y_train)
print(f"Train Score: {gbrt.score(X_train, y_train)}    Test Score: {gbrt.score(X_test, y_test)}")
