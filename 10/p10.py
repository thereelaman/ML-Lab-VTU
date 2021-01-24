import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def kernel(point, xmat, k):
    m,n = np.shape(xmat)
    weights = np.mat(np.eye((m))) # eye - identity matrix
    for j in range(m):
        diff = point - X[j]
        weights[j,j] = np.exp(diff*diff.T/(-2.0*k**2))
    return weights

def localWeight(point,xmat,ymat,k):
    wt = kernel(point,xmat,k)
    W = (X.T*(wt*X)).I*(X.T*(wt*ymat.T))
    return W

def localWeightRegression(xmat,ymat,k):
    m,n = np.shape(xmat)
    ypred = np.zeros(m)
    for i in range(m):
        ypred[i] = xmat[i]*localWeight(xmat[i],xmat,ymat,k)
    return ypred

# load data points
data = pd.read_csv('10/10data_tips.csv') 
bill = np.array(data.total_bill) # We use only Bill amount and Tips data
tip = np.array(data.tip)
mbill = np.mat(bill) # .mat will convert nd array is converted in 2D array
mtip = np.mat(tip)
m = np.shape(mtip)[1]
one = np.ones((1,m), dtype= int)
X = np.hstack((one.T,mbill.T)) # 244 rows, 2 cols

ypred = localWeightRegression(X,mtip,8) # increase k to get smooth curves

xsort = X.copy()
xsort.sort(axis=0)
plt.scatter(bill, tip, color='blue')
plt.plot(xsort[:, 1], ypred[X[:, 1].argsort(0)], color='black', linewidth=5)
plt.xlabel('Total Bill')
plt.ylabel('Tip')
plt.show()