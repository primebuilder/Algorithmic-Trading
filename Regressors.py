from sklearn.externals import joblib
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from matplotlib import pyplot as plt
import pickle
import os
import numpy as np
import pandas as pd
import itertools

SEED = 100
random_state = np.random.RandomState(SEED)

def MSE(predicted, test_label):
    return ((predicted-test_label)**2).mean()

def linear_regressor(train_data,train_label,test_data,test_label,parameters):
    regr = linear_model.LinearRegression()
    regr.fit(train_data, train_label)
    score = regr.score(test_data, test_label)
    predict = regr.predict(test_data)
    mse = MSE(predict, test_label)
    print 'MSE '+parameters+' '+ str(mse[0])
    df = pd.Series(predict.flatten(),index=test_label.index)
    price = train_label.append(test_label)
    plt.title('Linear Regression on '+ parameters)
    plt.plot(price[1000:-1],label='actual price')
    plt.plot(df,label='predicted price')
    plt.legend(loc='lower right')
    plt.xlabel('Dates')
    plt.ylabel('Price')
    # plt.show()
    directory = './linear/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory + parameters + '.png')
    plt.close()
    # print predict
    return score

def poly_regressor(train_data, train_label, test_data, test_label, parameters):

    min_error = 10000000000
    error = []
    deg = [2,3,4,5,6]
    degr = 2
    # pca = PCA(n_components=15)
    # train_data = pca.fit_transform(train_data)
    # test_data = pca.fit_transform(test_data)
    for i in range(2,7):
        poly = PolynomialFeatures(degree=i)
        train_data_ = poly.fit_transform(train_data)
        test_data_ = poly.fit_transform(test_data)
        clf = linear_model.LinearRegression()
        clf.fit(train_data_, train_label)
        predict = clf.predict(test_data_)
        mse = MSE(predict,test_label)
        error.append(mse)
        # print mse[0]
        if( mse[0] < min_error ):
            min_error = mse[0]
            degr = i
    # plt.title('MSE vs Degree')
    # plt.plot(deg,error)
    # plt.xlabel('Degree')
    # plt.ylabel('MSE')
    # directory = './poly/'
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    # plt.savefig(directory + 'MSE' + parameters + '.png')
    # plt.close()

    poly = PolynomialFeatures(degree=degr)
    train_data_ = poly.fit_transform(train_data)
    test_data_ = poly.fit_transform(test_data)
    clf = linear_model.LinearRegression()
    clf.fit(train_data_, train_label)
    predict = clf.predict(test_data_)
    mse = MSE(predict, test_label)
    print 'MSE '+parameters+' '+str(mse[0])
    score = clf.score(test_data_, test_label)
    df = pd.Series(predict.flatten(), index=test_label.index)
    price = train_label.append(test_label)
    plt.title('Polynomial Regression on ' + parameters)
    plt.plot(price[1000:-1],label='actual price')
    plt.plot(df,label='predicted price')
    plt.legend(loc='lower right')
    plt.xlabel('Dates')
    plt.ylabel('Price')
    # plt.show()
    directory = './poly/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory + parameters + '.png')
    plt.close()
    # print predict
    return score

def ridge_regressor(train_data, train_label, test_data, test_label, parameters):
    min_error = 10000000000
    error = []
    alp = [1e-4,1e-3,1e-2,1e-1,1,10,100,1000,10000]
    fin_alpha = 0
    for i in range(0, len(alp)):
        regr = linear_model.Ridge(alpha=alp[i],random_state=random_state)
        regr.fit(train_data, train_label)
        predict = regr.predict(test_data)
        mse = MSE(predict, test_label)
        error.append(mse)
        # print mse[0]
        if (mse[0] < min_error):
            min_error = mse[0]
            fin_alpha = alp[i]
    plt.title('MSE vs Alpha')
    plt.plot(alp, error)
    plt.xscale('log')
    plt.xlabel('Alpha')
    plt.ylabel('MSE')
    directory = './ridge/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory + 'MSE' + parameters + '.png')
    plt.close()
    regr = linear_model.Ridge(alpha= fin_alpha,random_state=random_state)
    regr.fit(train_data, train_label)
    score = regr.score(test_data, test_label)
    predict = regr.predict(test_data)
    mse = MSE(predict, test_label)
    print 'MSE '+parameters+' '+ str(mse[0])
    df = pd.Series(predict.flatten(), index=test_label.index)
    price = train_label.append(test_label)
    plt.title('Ridge Regression on ' + parameters)
    plt.plot(price[1000:-1],label='actual price')
    plt.plot(df, label='predicted price')
    plt.legend(loc='lower right')
    plt.xlabel('Dates')
    plt.ylabel('Price')
    # plt.show()
    directory = './ridge/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory + parameters + '.png')
    plt.close()
    # print predict
    return score

def lasso_regressor(train_data, train_label, test_data, test_label, parameters):
    min_error = 10000000000
    error = []
    alp = [1e-4,1e-3,1e-2, 1e-1, 1, 10, 100, 1000, 10000]
    fin_alpha = 0
    for i in range(0, len(alp)):
        regr = linear_model.Lasso(alpha=alp[i], max_iter=10000,random_state=random_state)
        regr.fit(train_data, train_label)
        predict = regr.predict(test_data)
        predict = map(lambda x:[x],predict)
        mse = MSE(np.array(predict), test_label)
        error.append(mse)
        # print mse[0]
        if (mse[0] < min_error):
            min_error = mse[0]
            fin_alpha = alp[i]
    plt.title('MSE vs Alpha')
    plt.plot(alp, error)
    plt.xscale('log')
    plt.xlabel('Alpha')
    plt.ylabel('MSE')
    directory = './lasso/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory + 'MSE' + parameters + '.png')
    plt.close()
    regr = linear_model.Lasso(alpha=fin_alpha,max_iter=10000,random_state=random_state)
    regr.fit(train_data, train_label)
    score = regr.score(test_data, test_label)
    predict = regr.predict(test_data)
    predict = map(lambda x: [x], predict)
    predict = np.array(predict)
    mse = MSE(np.array(predict), test_label)
    print 'MSE ' + parameters + ' ' + str(mse[0])
    df = pd.Series(predict.flatten(), index=test_label.index)
    price = train_label.append(test_label)
    plt.title('Lasso Regression on ' + parameters)
    plt.plot(price[1000:-1],label='actual price')
    plt.plot(df,label='predicted price')
    plt.legend(loc='lower right')
    plt.xlabel('Dates')
    plt.ylabel('Price')
    # plt.show()
    directory = './lasso/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory + parameters + '.png')
    plt.close()
    # print predict
    return score

def elasticnet_regressor(train_data, train_label, test_data, test_label, parameters):
    min_error = 10000000000
    error = []
    alp = [1e-4,1e-3,1e-2, 1e-1, 1, 10, 100, 1000, 10000]
    fin_alpha = 0
    for i in range(0, len(alp)):
        regr = linear_model.ElasticNet(alpha=alp[i],random_state=random_state)
        regr.fit(train_data, train_label)
        predict = regr.predict(test_data)
        predict = map(lambda x: [x], predict)
        mse = MSE(np.array(predict), test_label)
        error.append(mse)
        # print mse[0]
        if (mse[0] < min_error):
            min_error = mse[0]
            fin_alpha = alp[i]
    plt.title('MSE vs Alpha')
    plt.plot(alp, error)
    plt.xscale('log')
    plt.xlabel('Alpha')
    plt.ylabel('MSE')
    directory = './elasticnet/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory + 'MSE' + parameters + '.png')
    plt.close()
    regr = linear_model.ElasticNet(alpha=fin_alpha,random_state=random_state)
    regr.fit(train_data, train_label)
    score = regr.score(test_data, test_label)
    predict = regr.predict(test_data)
    predict = map(lambda x: [x], predict)
    predict = np.array(predict)
    mse = MSE(np.array(predict), test_label)
    print 'MSE ' + parameters + ' ' + str(mse[0])
    df = pd.Series(predict.flatten(), index=test_label.index)
    price = train_label.append(test_label)
    plt.title('Elastic-net Regression on ' + parameters)
    plt.plot(price[1000:-1],label='actual price')
    plt.plot(df,label='predicted price')
    plt.legend(loc='lower right')
    plt.xlabel('Dates')
    plt.ylabel('Price')
    # plt.show()
    directory = './elasticnet/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory+parameters+'.png')
    plt.close()
    # print predict
    return score

def adaboost_regressor(train_data, train_label, test_data, test_label, parameters):
    min_error = 10000000000
    error = []
    learn_rate = [1e-2,1e-1,1,10,100,500,1000]
    n_est = [20,40,60,80,100]
    comb = list(itertools.product(learn_rate, n_est))
    # print comb
    fin_learn = 0
    fin_est = 0
    for i in range(0,len(comb)):
        regr = AdaBoostRegressor(n_estimators=comb[i][1],learning_rate=comb[i][0],random_state=random_state)
        regr.fit(train_data, train_label)
        predict = regr.predict(test_data)
        predict = map(lambda x: [x], predict)
        mse = MSE(np.array(predict), test_label)
        error.append(mse)
        # print mse[0]
        if (mse[0] < min_error):
            min_error = mse[0]
            # print comb[i]
            fin_learn = comb[i][0]
            fin_est = comb[i][1]
        else:
            continue

    plt.figure(figsize=(10, 12))
    plt.title('MSE vs (learning rate, n_estimate)')
    plt.plot(range(len(comb)), error)
    plt.xticks(np.arange(len(comb)),comb,rotation=90)
    plt.xlabel('(learning rate, n_estimate)')
    plt.ylabel('MSE')
    directory = './adaboost/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory + 'MSE' + parameters + '.png')
    plt.close()
    regr = AdaBoostRegressor(n_estimators=80, learning_rate=1,random_state=random_state)
    regr.fit(train_data, train_label)
    score = regr.score(test_data,test_label)
    predict = regr.predict(test_data)
    predict = map(lambda x: [x], predict)
    predict = np.array(predict)
    mse = MSE(np.array(predict), test_label)
    print 'MSE ' + parameters + ' ' + str(mse[0])
    df = pd.Series(predict.flatten(), index=test_label.index)
    price = train_label.append(test_label)
    plt.title('AdaBoost on ' + parameters)
    plt.plot(price[1000:-1],label='actual price')
    plt.plot(df,label='predicted price')
    plt.legend(loc='lower right')
    plt.xlabel('Dates')
    plt.ylabel('Price')
    # plt.show()
    directory = './adaboost/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory + parameters + '.png')
    plt.close()
    return score

def svm_regressor(train_data, train_label, test_data, test_label, parameters):
    min_error = 10000000000
    error = []

    # tuned_parameters = [{'kernel': ['rbf'], 'gamma': [100,10,1,1e-1, 1e-2,],
    #                      'C': [0.1,1, 10, 100], 'epsilon':[ 100, 1000, 10000,1e6,1e8]}]
    #                     # {'kernel': ['linear'], 'C': [1, 10, 100, 1000], 'epsilon': [1, 10,100,1000]},
    #                     # {'kernel':['poly'],'gamma': [1e-3, 1e-4],
    #                     #  'C': [1, 10, 100, 1000], 'epsilon':[ 1, 10, 100,1000]}]
    # # {'kernel': ['linear'], 'C': [1, 10, 100, 1000], 'epsilon': [1e-2, 1e-1, 1, 10]}
    # clf = GridSearchCV(SVR(), tuned_parameters, cv=5,verbose=1,n_jobs=-1)
    # clf.fit(train_data, train_label)
    # print clf.best_params_
    # print clf.cv_results_
    # tuned_parameters = [{'C': [1e-2,1e-1,1, 10, 100], 'epsilon': [1, 10, 100, 1000,10000]}]
    # clf = GridSearchCV(LinearSVR(random_state=random_state), tuned_parameters, cv=5, verbose=1, n_jobs=-1)
    # clf.fit(train_data, train_label)
    # print clf.best_params_
    # print clf.cv_results_

    # regr = SVR(kernel='rbf', gamma=0.01,C=100)
    # regr.fit(train_data, train_label)
    # score = regr.score(test_data, test_label)
    # predict = regr.predict(test_data)
    # predict = map(lambda x: [x], predict)
    # predict = np.array(predict)
    # mse = MSE(np.array(predict), test_label)
    # if (mse[0] < min_error):
    #     min_error = mse[0]
    # print mse[0]
    regr = LinearSVR(C=0.001,epsilon=1, random_state=random_state)
    regr.fit(train_data, train_label)
    score = regr.score(test_data, test_label)
    predict = regr.predict(test_data)
    predict = map(lambda x: [x], predict)
    predict = np.array(predict)
    mse = MSE(np.array(predict), test_label)
    if (mse[0] < min_error):
        min_error = mse[0]

    print 'MSE ' + parameters + ' ' + str(mse[0])

    df = pd.Series(predict.flatten(), index=test_label.index)
    price = train_label.append(test_label)
    plt.title('SVM Regression on ' + parameters)
    plt.plot(price[1000:-1],label='actual price')
    plt.plot(df,label='predicted price')
    plt.legend(loc='lower right')
    plt.xlabel('Dates')
    plt.ylabel('Price')
    # plt.show()
    directory = './svm/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory + parameters + '.png')
    plt.close()
    return

def randomforest_regressor(train_data, train_label, test_data, test_label, parameters):
    min_error = 10000000000
    error = []
    n_est = [20, 40, 60, 80, 100, 120, 140]
    max_feature = ['auto','log2','sqrt']
    final_n_est = 0
    final_max_feature = ''
    comb = list(itertools.product(max_feature, n_est))
    for i in range(0,len(comb)):
        regr = RandomForestRegressor(n_estimators=comb[i][1], max_features=comb[i][0], random_state=random_state)
        regr.fit(train_data, train_label)
        predict = regr.predict(test_data)
        predict = map(lambda x: [x], predict)
        mse = MSE(np.array(predict), test_label)
        error.append(mse)
        # print mse[0]
        if (mse[0] < min_error):
            min_error = mse[0]
            final_n_est = comb[i][1]
            final_max_feature = comb[i][0]
    plt.figure(figsize=(10, 12))
    plt.title('MSE vs (max_features , n_estimate)')
    plt.plot(range(len(comb)), error)
    plt.xticks(np.arange(len(comb)), comb, rotation=90)
    plt.xlabel('(max_features, n_estimate)')
    plt.ylabel('MSE')
    directory = './randomforest/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory + 'MSE' + parameters + '.png')
    plt.close()
    regr = RandomForestRegressor(n_estimators=final_n_est, max_features=final_max_feature, random_state=random_state)
    regr.fit(train_data, train_label)
    predict = regr.predict(test_data)
    score = regr.score(test_data,test_label)
    predict = map(lambda x: [x], predict)
    predict = np.array(predict)
    mse = MSE(predict , test_label)
    print 'MSE ' + parameters + ' ' + str(mse[0])
    df = pd.Series(predict.flatten(), index=test_label.index)
    price = train_label.append(test_label)
    plt.title('RandomForest Regression on ' + parameters)
    plt.plot(price[1000:-1],label='actual price')
    plt.plot(df,label='predicted price')
    plt.legend(loc='lower right')
    plt.xlabel('Dates')
    plt.ylabel('Price')
    # plt.show()
    directory = './randomforest/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory + parameters + '.png')
    plt.close()
    return score