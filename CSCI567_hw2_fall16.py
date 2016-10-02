#@author Sagar Makwana
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import numpy as np
import heapq
import operator

#-----------------------------------------Function Definition -----------------------------------------------

#This method calculates the gradient update over all
def gradient_update(x,y,n,w):
    gradient = np.zeros(np.size(x,1))
    alpha = 0.2/n
    for i in range(0,n):
        xi = x[i,:]
        gradient = gradient + (np.dot(xi,w) - y[i])*xi

    w = w - alpha*gradient

    return w

def ridge_gradient_update(x,y,n,w,regLambda):
    gradient = np.zeros(np.size(x,1))
    alpha = 0.2/n
    for i in range(0,n):
        xi = x[i,:]
        gradient = gradient + (np.dot(xi,w) - y[i])*xi

    w = w - alpha*(gradient + n*regLambda*w)

    return w

#This function calculates the mean squared error of the input samples
def mean_squared_loss(x,y,n,w):
    return np.sum(np.square(np.dot(x,w) - y))/n

# This function calculates the analytical weights for linear regression
def get_linear_regression_weights(x,y):
    a1 = x.transpose().dot(x)
    a2 = np.linalg.inv(a1)
    a3 = a2.dot(x.transpose())
    a4 = a3.dot(y)

    return a4

# This function calculates the analytical weights for ridge regression
def get_ridge_regression_weights(x,y,regLambda):
    a0 = regLambda * np.eye(np.size(x,1))
    a1 = x.transpose().dot(x) + a0
    a2 = np.linalg.inv(a1)
    a3 = a2.dot(x.transpose())
    a4 = a3.dot(y)

    return  a4

def getMaxCorrelatedFeature(x,listFeatures,target):
    dict = {}
    for feature in listFeatures:
        dict[feature] = np.absolute(np.corrcoef(x[:,feature],target)[0][1])

    maxindex = max(dict.iteritems(), key=operator.itemgetter(1))[0]
    return maxindex


#---------------------------------------------- 3.1 Dataset ---------------------------------------------------
# Loading the Dataset
boston = load_boston()

# Training/Test Splitting
train_x = np.zeros((433,13))
train_y = np.zeros((433,1))
test_x = np.zeros((73,13))
test_y = np.zeros((73,1))
n = np.size(train_x,0)
test_n = np.size(test_x,0)

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
#for i in range(0,np.size(train_x,1)):
#    fig = plt.figure()
#    plt.hist(train_x[:,i],bins=10)
#    plt.title("Feature "+str(i+1))
#    plt.xlabel("Value")
#    plt.ylabel("Frequency")
#    fig.savefig('histogram_feature_'+str(i+1)+'.png')


#Pearson Correlation Coefficient
pcc = np.array(np.zeros(np.size(train_x,1)))
yi = train_y[:,0]
sum_yi = np.sum(yi)
sum_yi_square = np.sum(np.square(yi))

for i in range(0,np.size(train_x,1)):
    xi = train_x[:,i]
    sum_xi = np.sum(xi)
    sum_xi_square = np.sum(np.square(xi))
    sum_xiyi = np.sum(xi*yi)
    pcc[i]  = ((n*sum_xiyi - sum_xi*sum_yi)/(np.sqrt(n*sum_xi_square - np.square(sum_xi)) * np.sqrt(n*sum_yi_square - np.square(sum_yi))))

print "Pearson Correlation Coefficients are as follows:"
for i in range(0,len(pcc)):
    print 'Attribute ' + str(i+1) + ' : ' + str(pcc[i])

#Data Preprocessing
#Normalizing Training data..
for i in range(0,np.size(train_x,1)):
    xi = train_x[:,i]
    mean = np.mean(xi)
    std = np.std(xi)
    train_x[:,i] = (xi - mean)*1.0/std;

#Normalizing Test data..
for i in range(0,np.size(test_x,1)):
    xi = test_x[:,i]
    mean = np.mean(xi)
    std = np.std(xi)
    test_x[:,i] = (xi - mean)*1.0/std;



#---------------------------------------------3.2 Linear Regression  ----------------------------------------------------


#Linear Regression
linear_train_x  = np.insert(train_x,0,np.ones(n),axis = 1) # Added bias term in the first column
linear_train_y = train_y[:,0]
linear_test_x  = np.insert(test_x,0,np.ones(test_n),axis=1)
linear_test_y = test_y[:,0]


print '\nLinear Regression Results:'
weights = get_linear_regression_weights(linear_train_x,linear_train_y)
loss = mean_squared_loss(linear_train_x,linear_train_y,n,weights)
print 'MSE for training Data: ' + str(loss)
loss = mean_squared_loss(linear_test_x,linear_test_y,test_n,weights)
print 'MSE for test Data: ' + str(loss)


#Ridge Regression

print '\nRidge Regression Calculations:'
regLambdas = [0.01,0.1,1.0]
for regLambda in regLambdas:
    print 'For lambda = ' + str(regLambda) + ' :'
    weights = get_ridge_regression_weights(linear_train_x,linear_train_y,regLambda)
    loss = mean_squared_loss(linear_train_x,linear_train_y,n,weights)
    print 'MSE for training data: ' + str(loss)
    loss = mean_squared_loss(linear_test_x,linear_test_y,test_n,weights)
    print 'MSE for test data: ' + str(loss)

#Cross validation for Ridge Regression
print '\nCross Validation results :'
regLambdas = [0.0001,0.001,0.01,0.1,1,10]
lossArray = np.array(np.zeros(len(regLambdas)))
k =int(n/10) #No. of samples in a single fold
index = 0
for regLambda in regLambdas:
    loss = 0
    for i in range(0,10):
        if i == 0:
            test_samples_x = linear_train_x[:(i+1)*k,:]
            test_samples_y = linear_train_y[:(i+1)*k]
            train_samples_x = linear_train_x[(i+1)*k:,:]
            train_samples_y = linear_train_y[(i+1)*k:]
        elif i == 9:
            test_samples_x = linear_train_x[i*k:,:]
            test_samples_y = linear_train_y[i*k:]
            train_samples_x = linear_train_x[:i*k,:]
            train_samples_y = linear_train_y[:i*k]
        else:
            test_samples_x = linear_train_x[i*k:(i+1)*k,:]
            test_samples_y = linear_train_y[i*k:(i+1)*k]
            train_samples_x = np.concatenate((linear_train_x[:i*k,:],linear_train_x[(i+1)*k:,:]),axis=0)
            train_samples_y = np.concatenate((linear_train_y[:i*k],linear_train_y[(i+1)*k:]),axis=0)

        weights = get_ridge_regression_weights(train_samples_x,train_samples_y,regLambda)
        loss = loss + mean_squared_loss(test_samples_x,test_samples_y,np.size(test_samples_x,0),weights)

    loss  = loss/10
    lossArray[index] = loss
    index = index + 1
    print 'Lambda = ' + str(regLambda) + ', Average MSE over 10 folds = ' + str(loss)

index = np.argmin(lossArray)  #index of the lambda with min average MSE

weights = get_ridge_regression_weights(linear_train_x,linear_train_y,regLambdas[index])
loss = mean_squared_loss(linear_test_x,linear_test_y,test_n,weights)
print 'MSE for test data with lambda='+ str(regLambdas[index])+ ' is : ' + str(loss)



#--------------------------------------3.3 Feature Selection ------------------------------------------

#3.3 a)
#At this point we have pcc which is the calculated pearson correlation array for training data
#Also we remove the bias element from the linear train x

linear_train_x = train_x
linear_train_y = train_y[:,0]
linear_test_x = test_x
linear_test_y = test_y[:,0]

max_indices = heapq.nlargest(4, range(len(pcc)), abs(pcc).take)
reduced_feature_x = linear_train_x[:,max_indices]
reduced_feature_x = np.insert(reduced_feature_x,0,np.ones(n),axis = 1)
reduced_test_x = linear_test_x[:,max_indices]
reduced_test_x = np.insert(reduced_test_x,0,np.ones(test_n),axis = 1)

print '\nFeature Selection:Scheme (a) results:'
print 'Attribute\tCorrelation'
for index in max_indices:
    print str(index+1) + '\t' + str(pcc[index])

weights = get_linear_regression_weights(reduced_feature_x,linear_train_y)
loss = mean_squared_loss(reduced_feature_x,linear_train_y,n,weights)
print 'MSE for training data: ' + str(loss)
loss = mean_squared_loss(reduced_test_x,linear_test_y,test_n,weights)
print 'MSE for test Data: ' + str(loss)


#3.3 b)
leftoverFeatures = range(0,13)
addedFeatures = []
residual = linear_train_y

print '\nFeature Selection:Scheme (b) results:'
print 'Attribute Selected:'
for i in range(0,4):
    maxFeatureIndex = getMaxCorrelatedFeature(train_x,leftoverFeatures,residual)
    print (maxFeatureIndex+1)
    addedFeatures.append(maxFeatureIndex)
    leftoverFeatures.remove(maxFeatureIndex)
    reduced_feature_x = train_x[:,addedFeatures]
    reduced_feature_x = np.insert(reduced_feature_x,0,np.ones(n),axis = 1)
    weights = get_linear_regression_weights(reduced_feature_x,linear_train_y)
    residual = linear_train_y - np.dot(reduced_feature_x,weights)

loss = mean_squared_loss(reduced_feature_x,linear_train_y,n,weights)
print 'MSE for training data: ' + str(loss)
reduced_test_x = test_x[:,addedFeatures]
reduced_test_x  = np.insert(reduced_test_x,0,np.ones(test_n),axis = 1)
loss = mean_squared_loss(reduced_test_x,linear_test_y,test_n,weights)
print 'MSE for test Data: ' + str(loss)


#Selection with brute force search

print '\nFeature Selection brute force results:'
print 'Attribute Selected:'
dict = {}
for i in range(0,np.size(train_x,1)):
    for j in range(i+1,np.size(train_x,1)):
        for k in range(j+1,np.size(train_x,1)):
            for l in range(k+1,np.size(train_x,1)):
                reduced_feature_x = train_x[:,[i,j,k,l]]
                reduced_feature_x = np.insert(reduced_feature_x,0,np.ones(n),axis = 1)
                weights = get_linear_regression_weights(reduced_feature_x,linear_train_y)
                loss = mean_squared_loss(reduced_feature_x,linear_train_y,n,weights)
                dict[i,j,k,l] = loss

addedFeatures = min(dict.iteritems(), key=operator.itemgetter(1))[0]

for i in addedFeatures:
    print i+1

reduced_feature_x = train_x[:,addedFeatures]
reduced_feature_x = np.insert(reduced_feature_x,0,np.ones(n),axis = 1)
weights = get_linear_regression_weights(reduced_feature_x,linear_train_y)
loss = mean_squared_loss(reduced_feature_x,linear_train_y,n,weights)
print 'MSE for training data: ' + str(loss)
reduced_test_x = test_x[:,addedFeatures]
reduced_test_x  = np.insert(reduced_test_x,0,np.ones(test_n),axis = 1)
loss = mean_squared_loss(reduced_test_x,linear_test_y,test_n,weights)
print 'MSE for test Data: ' + str(loss)


#-----------------------------------3.4 Polynomial feature expansion ------------------------------

poly_train_x = np.array(np.zeros((np.size(train_x,0),91)))
index = 0
for i in range(0,13):
    for j in range(i,13):
        poly_train_x[:,index] = train_x[:,i]*train_x[:,j]
        index = index + 1

for i in range(0,np.size(poly_train_x,1)): #Normalize
    xi = poly_train_x[:,i]
    mean = np.mean(xi)
    std = np.std(xi)
    print i,mean,std
    poly_train_x[:,i] = (xi - mean)*1.0/std;

poly_test_x = np.array(np.zeros((np.size(test_x,0),91)))
index  = 0
for i in range(0,13):
    for j in range(i,13):
        poly_test_x[:,index] = np.multiply(test_x[:,i],test_x[:,j])
        index = index + 1

for i in range(0,np.size(poly_test_x,1)): #Normalize
    xi = poly_test_x[:,i]
    mean = np.mean(xi)
    std = np.std(xi)
    poly_test_x[:,i] = (xi - mean)*1.0/std;

poly_train_x = np.hstack([train_x,poly_train_x])
poly_train_x = np.insert(poly_train_x,0,np.ones(n),axis = 1)
weights = get_linear_regression_weights(poly_train_x,linear_train_y)
loss = mean_squared_loss(poly_train_x,linear_train_y,n,weights)
print 'MSE for training data: ' + str(loss)

poly_test_x = np.hstack([test_x,poly_test_x])
poly_test_x  = np.insert(poly_test_x,0,np.ones(test_n),axis = 1)
loss = mean_squared_loss(poly_test_x,linear_test_y,test_n,weights)
print 'MSE for test Data: ' + str(loss)