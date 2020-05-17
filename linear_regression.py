######################################################################
#    Régression linéaire avec descente de gradient stochastique
#               Author :  Youssef Kartit
######################################################################

''' Module régression linéaire'''

##### Libraries :
from math import sqrt
import dataprep as dp

#### Métrique et evaluation d'algorithme
def rmse(test, predicted):
    ''' This function computes  the root mean squared error (rmse) '''
    sum_error = 0.0
    for i in range(len(test)):
        prediction_error = predicted[i] - test[i]
        sum_error += prediction_error**2
    mean_err = (1 / len(test)) * sum_error
    return sqrt(mean_err)

#### Prédiction de la valeur à partir d'un x
def predict(x, w):
    ''' This fuction computes the prediction y based on a new value x and w '''
    y_ = w[0]
    for k in range(len(x)-1):
        y_ += w[k + 1] * x[k]
    return y_
#### Evaluer un algorithme :
def evaluate_algorithm(dataset, algorithm, K, *args):
    ''' This function '''
    folds = dp.crossvalidation_split(dataset, K)
    scores = []
    for fold in folds: 
        trainset = list(folds)
        trainset.remove(fold)
        trainset = sum(trainset, [])
        testset = []
        for row in fold:
            row_copy = list(row)
            testset.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(trainset, testset, *args)
        test = [row[-1] for row in fold]
        r_mse = rmse(test, predicted)
        scores.append(r_mse)
    return scores

#### coeff 
def coefficients_sgd(train, learning_rate, T):
    ''' This function compute the optimal w'''
    w_ = [0 for i in range(len(train[0]))] 
    for t in range(T):
        sum_error = 0
        for row in train: 
            y_ = predict(row, w_)
            v = y_ - row[-1]
            sum_error += v**2
            w_[0] = w_[0] - learning_rate * v 
            for k in range(len(row)-1):
                w_[k + 1] = w_[k + 1] - learning_rate * v * row[k]  
        learning_rate = learning_rate / ((t+1)**(1/4))
    return w_

#### Régression linéaire avec SGD: 
def linear_regression_sgd(train, test, learning_rate, T):
    predictions = []
    w_ = coefficients_sgd(train, learning_rate, T)
    for k in range(len(test)):
        y_ = predict(test[k], w_)
        predictions.append(y_)
    return predictions