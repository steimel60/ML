#A doc containing simple code as a means to introduce the reader to different
#libraries commonly used for Machine Learning


#--------------------- NumPy ---------------------#
import numpy as np

x = np.array([[1,2,3],[4,5,6]])
#print(x)


#--------------------- SciPy ---------------------#
from scipy import sparse #arrays that are mostly zeros

eye = np.eye(4) #special type of np array with 1s on diagonal
#print(eye)
sparse_matrix = sparse.csr_matrix(eye) #convert np array to sparse array
#print(sparse_matrix) #prints ""(row,col) val" of non-zero values


#------------------- Matplotlib -------------------#
import matplotlib.pyplot as plt

x = np.linspace(-10,10,100) #Creates a seq of 100 numbers between -10 and 10
y = np.sin(x) #array of sin(x)
plt.plot(x,y,marker='x') #Line graph with x at each point
#plt.show()

#--------------------- Pandas ---------------------#
import pandas as pd

#Create dict holding data
data = {'Name': ["Daedan", "Lexron", "Kcin", "Dyffros"],
        'Location': ["Minis Vale", "Shioso", "Zysteria", "Osaunu"],
        'Age': [20,24,21,19]
        }

df = pd.DataFrame(data) #Store dict in pandas DataFrame
print(df,'\n')
print(df[df.Age<21]) #get rows where data meets certain conditon
