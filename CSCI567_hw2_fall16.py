#@author Sagar Makwana
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import numpy as np

# 3.1 Dataset
# Loading the Dataset
boston = load_boston()

# Training/Test Splitting
train_x = np.zeros((433,13))
train_y = np.zeros((433,1))
test_x = np.zeros((73,13))
test_y = np.zeros((73,1))

test_counter = 0
train_counter = 0
for i in range(0,len(boston['data'])):
    if i%7 == 0:
        test_x[test_counter] = (boston['data'][i])
        test_y[test_counter] = (boston['target'][i])
        test_counter = test_counter + 1
    else:
        train_x[train_counter] = boston['data'][i]
        train_y[train_counter] = boston['target'][i]
        train_counter = train_counter + 1


#Data Analysis

#Histogram
for i in range(0,np.size(train_x,1)):
    fig = plt.figure()
    plt.hist(train_x[:,i],bins=10)
    plt.title("Feature "+str(i+1))
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    fig.savefig('histogram_feature_'+str(i+1)+'.png')


#Pearson Correlation Coefficient
pcc = []
yi = train_y[:,0]
sum_yi = np.sum(yi)
sum_yi_square = np.sum(np.square(yi))
n = np.size(train_x,0)

for i in range(0,np.size(train_x,1)):
    xi = train_x[:,i]
    sum_xi = np.sum(xi)
    sum_xi_square = np.sum(np.square(xi))
    sum_xiyi = np.sum(xi*yi)
    pcc.append((n*sum_xiyi - sum_xi*sum_yi)/(np.sqrt(n*sum_xi_square - np.square(sum_xi)) * np.sqrt(n*sum_yi_square - np.square(sum_yi))))

print "Pearson Correlation Coefficients are as follows:"
for i in range(0,len(pcc)):
    print 'Attribute ' + str(i+1) + ' : ' + str(pcc[i])

#Data Preprocessing
for i in range(0,np.size(train_x,1)):
    xi = train_x[:,i]
    mean = np.mean(xi)
    std = np.std(xi)
    train_x[:,i] = (xi - mean)*1.0/std;



