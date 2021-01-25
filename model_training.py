#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Sat May 16 01:48:18 2020

@author: roberto
'''

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, roc_curve, auc, plot_roc_curve
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn import svm, tree
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split, StratifiedKFold
from sklearn.feature_selection import SelectFromModel


# from datetime import datetime
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import MinMaxScaler


# =============================================================================
# noise_strategy = {avg, default, remove_incomplete}
# default_value = [0, 1]
# balance_strategy = {undersample_mayority, oversample_minority, None}
# estates = ['BAJA_CALIFORNIA', 'TLAXCALA', 'SONORA', 'COLIMA', 'CDMX']
# extra_symptoms = True  #Anosmia & Dysgeusia
# =============================================================================

def preprocess_data(noise_strategy='remove_incomplete', default_value=0, balance_strategy='undersample_mayority', estates = ['BAJA_CALIFORNIA', 'TLAXCALA', 'SONORA', 'COLIMA', 'CDMX'], extra_symptoms = True):
    rawDs = pd.read_csv('covid-19_tcsbc.csv').sample(frac=1, random_state=1)
    # rawDs = pd.read_csv('tlaxcala_17-06-2020_raw.csv').sample(frac=1, random_state=1)
    # rawDs.FINAL_RESULT.unique()
    
    #Filter estates
    rawDs = rawDs.loc[rawDs['ESTATE'].isin(estates)]
    
    #Filter Extra Symptoms
    #Dtaset Symptoms, contacto y subito
    if (extra_symptoms):
        ds = pd.concat([rawDs.iloc[:, 0], rawDs.iloc[:, 16:37], rawDs.iloc[:, [47, 48]]], axis=1) #Merged 5 States
    else:
        ds = pd.concat([rawDs.iloc[:, 0], rawDs.iloc[:, 16:35], rawDs.iloc[:, [47, 48]]], axis=1) #Merged 5 States
    # ds = pd.concat([rawDs.iloc[:, 33:54], rawDs.iloc[:, [69, 80]]], axis=1) #Tlaxcala 17-Jun
    
    
    ds = pd.concat([ds.loc[ds['FINAL_RESULT'] == 'NEGATIVE', :], ds.loc[ds['FINAL_RESULT'] == 'SARS-CoV-2', :]], axis=0)
    ds = ds.replace('NO', 0)
    ds = ds.replace('YES', 1)
    ds = ds.replace('NEGATIVE', 0)
    ds = ds.replace('SARS-CoV-2', 1)
    
    #Noise strategy (filling the blanks)
    if (noise_strategy == 'avg'):
        #NaN y 'SE IGNORA' are set to the average of the column
        for column in ds:
            if (column != 'FINAL_RESULT'):
                colNoNoise = pd.concat([ds.loc[ds[column] == 1], ds.loc[ds[column] == 0]], axis=0)
                colSUM = colNoNoise.loc[ds[column] == 1, column].sum()
                colAVG = colSUM/colNoNoise.shape[0]
                ds[column] = ds[column].fillna(colAVG)
                ds[column] = ds[column].replace('UNKNOWN', colAVG)
    elif (noise_strategy == 'default'):
        #NaN y 'SE IGNORA' are set to a default value
        for column in ds:
            if (column != 'FINAL_RESULT'):
                ds[column] = ds[column].fillna(default_value)
                ds[column] = ds[column].replace('UNKNOWN', default_value)
    elif (noise_strategy == 'remove_incomplete'):
        #NaN y 'SE IGNORA' se eliminan
        ds = ds.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
        for column in ds:
            if (column != 'FINAL_RESULT'):
                ds = ds.loc[ds[column] != 'UNKNOWN']
    
    #Balancing dataset
    dsP = ds.loc[ds['FINAL_RESULT'] == 1, :]
    dsN = ds.loc[ds['FINAL_RESULT'] == 0, :]
    nPositivos = dsP.shape[0]
    nNegativos = dsN.shape[0]
    sampleP = 1
    sampleN = 1
    if (balance_strategy == 'undersample_mayority'):
        if (nPositivos > nNegativos):
            sampleP = nNegativos/nPositivos
        elif (nNegativos > nPositivos):
            sampleN = nPositivos/nNegativos
    elif (balance_strategy == 'oversample_minority'):
        if (nPositivos > nNegativos):
            sampleN = nNegativos/nPositivos
        elif (nNegativos > nPositivos):
            sampleP = nPositivos/nNegativos
    dsN = dsN.sample(frac=sampleN, random_state=1)
    dsP = dsP.sample(frac=sampleP, random_state=1)
    ds = pd.concat([dsP, dsN], axis=0).sample(frac=1, random_state=36).reset_index(drop=True)
    # ds.to_csv('clean_ds_train.csv')
    
    return ds

def dt_parameters():
    criterion = ['gini', 'entropy']
    splitter = ['best', 'random']
    max_depth = [int(x) for x in np.linspace(start = 2, stop = 103, num = 50)]
    min_samples_split = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    min_samples_leaf = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    # max_features = ['auto', 'sqrt', 'log2', None]
    max_features = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ccp_alpha = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    params = {'criterion' : criterion,
              'splitter' : splitter,
              'max_depth' : max_depth,
              'min_samples_split' : min_samples_split,
              'min_samples_leaf' : min_samples_leaf,
              'max_features' : max_features,
              'ccp_alpha' : ccp_alpha}
    perms = 2*2*50*9*9*10*11    
    return params, perms

def rf_parameters():
    n_estimators = [int(x) for x in np.linspace(start = 2, stop = 2003, num = 500)]
    criterion = ['gini', 'entropy']
    max_depth = [int(x) for x in np.linspace(start = 2, stop = 103, num = 50)]
    min_samples_split = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    min_samples_leaf = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    # max_features = ['auto', 'sqrt', 'log2', None]
    max_features = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    bootstrap = [True, False]
    n_jobs = [-1]
    ccp_alpha = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    params = {'n_estimators' : n_estimators,
              'criterion' : criterion,
              'max_depth' : max_depth,
              'min_samples_split' : min_samples_split,
              'min_samples_leaf' : min_samples_leaf,
              'max_features' : max_features,
              'bootstrap' : bootstrap,
              'n_jobs' : n_jobs,
              'ccp_alpha' : ccp_alpha}
    perms = 500*2*50*9*9*10*2*11    
    return params, perms

def svm_parameters():
    C = [float(x) for x in np.linspace(start = 0, stop = 2, num = 40)]
    kernel = ['linear', 'poly', 'rbf', 'sigmoid']
    degree = [1, 2, 3, 4, 5, 6]
    gamma = [float(x) for x in np.linspace(start = 0.000000001, stop = 0.999, num = 100)]
    coef0 = [float(x) for x in np.linspace(start = 0, stop = 1, num = 200)]
    shrinking = [True, False]
    probability = [True]
    params = {'C' : C,
              'kernel' : kernel,
              'degree' : degree,
              'gamma' : gamma,
              'coef0' : coef0,
              'shrinking' : shrinking,
              'probability' : probability}
    perms = 40*4*6*100*200*2    
    return params, perms

def mlp_parameters():
    hidden_layer_sizes = [(15), (50), (80), (100), (200),
                          (15, 15), (15, 50), (15, 80), (15, 100), (15, 200),
                          (50, 15), (50, 50), (50, 80), (50, 100), (50, 200),
                          (80, 15), (80, 50), (80, 80), (80, 100), (80, 200),
                          (100, 15), (100, 50), (100, 80), (100, 100), (100, 200),
                          (200, 15), (200, 50), (200, 80), (200, 100), (200, 200),
                          (15, 15, 15), (15, 15, 50), (15, 15, 80), (15, 15, 100), (15, 15, 200),
                          (15, 50, 15), (15, 50, 50), (15, 50, 80), (15, 50, 100), (15, 50, 200),
                          (15, 80, 15), (15, 80, 50), (15, 80, 80), (15, 80, 100), (15, 80, 200),
                          (15, 100, 15), (15, 100, 50), (15, 100, 80), (15, 100, 100), (15, 100, 200),
                          (15, 200, 15), (15, 200, 50), (15, 200, 80), (15, 200, 100), (15, 200, 200),
                          (50, 15, 15), (50, 15, 50), (50, 15, 80), (50, 15, 100), (50, 15, 200),
                          (50, 50, 15), (50, 50, 50), (50, 50, 80), (50, 50, 100), (50, 50, 200),
                          (50, 80, 15), (50, 80, 50), (50, 80, 80), (50, 80, 100), (50, 80, 200),
                          (50, 100, 15), (50, 100, 50), (50, 100, 80), (50, 100, 100), (50, 100, 200),
                          (50, 200, 15), (50, 200, 50), (50, 200, 80), (50, 200, 100), (50, 200, 200),
                          (80, 15, 15), (80, 15, 50), (80, 15, 80), (80, 15, 100), (80, 15, 200),
                          (80, 50, 15), (80, 50, 50), (80, 50, 80), (80, 50, 100), (80, 50, 200),
                          (80, 80, 15), (80, 80, 50), (80, 80, 80), (80, 80, 100), (80, 80, 200),
                          (80, 100, 15), (80, 100, 50), (80, 100, 80), (80, 100, 100), (80, 100, 200),
                          (80, 200, 15), (80, 200, 50), (80, 200, 80), (80, 200, 100), (80, 200, 200),
                          (100, 15, 15), (100, 15, 50), (100, 15, 80), (100, 15, 100), (100, 15, 200),
                          (100, 50, 15), (100, 50, 50), (100, 50, 80), (100, 50, 100), (100, 50, 200),
                          (100, 80, 15), (100, 80, 50), (100, 80, 80), (100, 80, 100), (100, 80, 200),
                          (100, 100, 15), (100, 100, 50), (100, 100, 80), (100, 100, 100), (100, 100, 200),
                          (100, 200, 15), (100, 200, 50), (100, 200, 80), (100, 200, 100), (100, 200, 200),
                          (200, 15, 15), (200, 15, 50), (200, 15, 80), (200, 15, 100), (200, 15, 200),
                          (200, 50, 15), (200, 50, 50), (200, 50, 80), (200, 50, 100), (200, 50, 200),
                          (200, 80, 15), (200, 80, 50), (200, 80, 80), (200, 80, 100), (200, 80, 200),
                          (200, 100, 15), (200, 100, 50), (200, 100, 80), (200, 100, 100), (200, 100, 200),
                          (200, 200, 15), (200, 200, 50), (200, 200, 80), (200, 200, 100), (200, 200, 200)]
    activation = ['identity', 'logistic', 'tanh', 'relu']
    solver = ['lbfgs', 'sgd', 'adam']
    alpha = [float(x) for x in np.linspace(start = 0.00001, stop = 0.001, num = 500)]
    learning_rate = ['constant', 'invscaling', 'adaptive']
    learning_rate_init = [float(x) for x in np.linspace(start = 0.00001, stop = 0.01, num = 20)]
    power_t = [float(x) for x in np.linspace(start = 0.1, stop = 1.0, num = 10)]
    shuffle = [True, False]
    params = {'hidden_layer_sizes' : hidden_layer_sizes,
              'activation' : activation,
              'solver' : solver,
              'alpha' : alpha,
              'learning_rate' : learning_rate,
              'learning_rate_init' : learning_rate_init,
              'power_t' : power_t,
              'shuffle' : shuffle}
    perms = 31*4*3*500*3*20*10*2    
    return params, perms

def plot_roc(model, X_test, y_test, title, subtitle):
    probs = model.predict_proba(X_test)
    preds = probs[:,1]
    fpr, tpr, threshold = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)
    
    plt.clf()
    plt.title('Receiver operating characteristic\n' + title)
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    fileName = 'figures/' + title + '_' + subtitle + '_roc.png'
    plt.savefig(fileName)
    plt.show()
    
    plt.clf()
    cm = plot_confusion_matrix(model, X_test, y_test, values_format='d', cmap='Blues', display_labels=['Negative', 'Positive'])
    cm.ax_.set_title('Confusion Matrix\n' + title)
    fileName = 'figures/' + title + '_' + subtitle + '_cm.png'
    plt.savefig(fileName)
    plt.show()
    return roc_auc

def k_fold_plot_roc(classifier, k, X, y, subtitle, file_identifier):
    # #############################################################################
    # Classification and ROC analysis
    
    # Run classifier with cross-validation and plot ROC curves
    cv = StratifiedKFold(n_splits=k)
    cv.get_n_splits(X, y)
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    cm = [[0 for x in range(2)] for y in range(2)]
    sensitivity = 0
    specificity = 0
    precision = 0
    f1 = 0
    accuracy = 0
    
    
    fig, ax = plt.subplots()
    # for train_index, test_index in skf.split(X, y):
    # for i, (train, test) in enumerate(cv.split(X, y)):
    for i, (train, test) in enumerate(cv.split(X, y)):
        classifier.fit(X.iloc[train], y.iloc[train])
        viz = plot_roc_curve(classifier,
                             X.iloc[test], y.iloc[test],
                             name='ROC fold {}'.format(i),
                             alpha=0.3, lw=1, ax=ax)
        # viz = plot_roc_curve(classifier, X.iloc[test], y.iloc[test], alpha=0.3, lw=1, ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
        cm = confusion_matrix(y.iloc[test],classifier.predict(X.iloc[test]))
        sensitivity += (cm[1][1]/(cm[1][0]+cm[1][1]))
        specificity += (cm[0][0]/(cm[0][0]+cm[0][1]))
        precision += (cm[1][1]/(cm[0][1]+cm[1][1]))
        sens = (cm[1][1]/(cm[1][0]+cm[1][1]))
        prec = (cm[0][0]/(cm[0][0]+cm[0][1]))
        f1 += 2*((sens*prec)/(sens+prec))
        accuracy += (cm[0][0]+cm[1][1])/(cm[0][0]+cm[1][1]+cm[0][1]+cm[1][0])
    
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
    title = 'Receiver operating characteristic example\n' + subtitle
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title=title)
    ax.legend(loc='lower right')
    
    # Here is the trick
    plt.gcf()
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = handles[10:]
    labels = labels[10:]
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    fileName = 'figures/' + subtitle + '_' + file_identifier + '_' + str(k) + 'fold_roc.png'
    plt.savefig(fileName)
    plt.show()
    
    sensitivity = sensitivity/k
    specificity = specificity/k
    precision = precision/k
    f1 = f1/k
    accuracy = accuracy/k
    clf_name = subtitle + '_' + file_identifier + '_' + str(k) + 'fold'
    # return ([clf_name, sensitivity, specificity, precision, f1, accuracy])
    return ({'classifier': [clf_name] , 'sensitivity': [sensitivity], 'specificity': [specificity], 'precision': [precision], 'f1': [f1], 'accuracy': [accuracy], 'mean_auc': [mean_auc], 'std_auc': [std_auc]})
    # print ('Sensitivity:%0.3f,Specificity:%0.3f,Precision:%0.3f,f1:%0.3f,Accuracy:%0.3f' % (sensitivity, specificity, precision, f1, accuracy))

def check_for_missing_values(estate_list):
    ds = preprocess_data(noise_strategy='None',
                         default_value=0,
                         balance_strategy='None',
                         estates = estate_list,
                         extra_symptoms = True)
    records_isna = ds.isna().sum()/ds.shape[0]
    records_missing = pd.DataFrame()
    for column in ds:
            if (column != 'FINAL_RESULT'):
                records_missing.loc[column, 0] = (len(ds.loc[ds[column] == 'UNKNOWN'])/ds.shape[0]) + records_isna.loc[column]
    return records_missing

def random_search(X, y, subtitle):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=36)
    
    # Random search for Decision Tree
    clf_dt = tree.DecisionTreeClassifier()
    params, perms = dt_parameters()
    n_iter = int(perms * 0.003)
    dt_random = RandomizedSearchCV(estimator = clf_dt,
                                   param_distributions = params,
                                   n_iter = n_iter,
                                   scoring = 'roc_auc',
                                   cv = 5,
                                   verbose=2,
                                   random_state=36,
                                   n_jobs = -1)
    search = dt_random.fit(X_train, y_train)
    scores_dt = pd.DataFrame(search.cv_results_)
    scores_dt.to_csv('scores/dt_scores_' + subtitle + '.csv')
    title = 'Decision Tree'
    plot_roc(dt_random.best_estimator_, X_test, y_test, title, subtitle)
    print(f'Best Parameters for Decision Tree: {search.best_params_}')
    
    # Random search for Random Forest
    clf_rf = RandomForestClassifier()
    params, perms = rf_parameters()
    n_iter = int(perms * 0.000006)
    rf_random = RandomizedSearchCV(estimator = clf_rf,
                                   param_distributions = params,
                                   n_iter = n_iter,
                                   scoring = 'roc_auc',
                                   cv = 5,
                                   verbose=2,
                                   random_state=36,
                                   n_jobs = -1)
    search = rf_random.fit(X_train, y_train)
    scores_rf = pd.DataFrame(search.cv_results_)
    scores_rf.to_csv('scores/rf_scores_' + subtitle + '.csv')
    title = 'Random Forest'
    plot_roc(rf_random.best_estimator_, X_test, y_test, title, subtitle)
    print(f'Best Parameters for Random Forest: {search.best_params_}')
    
    # Random search for MLP
    clf_mlp = MLPClassifier()
    params, perms = mlp_parameters()
    n_iter = int(perms * 0.000024)
    mlp_random = RandomizedSearchCV(estimator = clf_mlp,
                                   param_distributions = params,
                                   n_iter = n_iter,
                                   scoring = 'roc_auc',
                                   cv = 5,
                                   verbose=2,
                                   random_state=36,
                                   n_jobs = -1)
    search = mlp_random.fit(X_train, y_train)
    scores_mlp = pd.DataFrame(search.cv_results_)
    scores_mlp.to_csv('scores/mlp_scores_' + subtitle + '.csv')
    title = 'Multi-Layer Perceptron'
    plot_roc(mlp_random.best_estimator_, X_test, y_test, title, subtitle)
    print(f'Best Parameters for Multi-Layer Perceptron: {search.best_params_}')
    
    # Random search for SVM
    clf_svm = svm.SVC()
    params, perms = svm_parameters()
    n_iter = int(perms * 0.00014)
    svm_random = RandomizedSearchCV(estimator = clf_svm,
                                   param_distributions = params,
                                   n_iter = n_iter,
                                   scoring = 'roc_auc',
                                   cv = 5,
                                   verbose=2,
                                   random_state=36,
                                   n_jobs = -1)
    search = svm_random.fit(X_train, y_train)
    scores_svm = pd.DataFrame(search.cv_results_)
    scores_svm.to_csv('scores/svm_scores_' + subtitle + '.csv')
    title = 'Support Vector Machine'
    plot_roc(svm_random.best_estimator_, X_test, y_test, title, subtitle)
    print(f'Best Parameters for SVM: {search.best_params_}')

def random_search_voters(X, y, subtitle, clf_rf, clf_dt, clf_mlp, clf_svm):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=36)
    
    # Random search for Voting 2, 14 Feats dt, rf, svm, mlp
    estimators = [[('DT', clf_dt), ('RF', clf_rf)], # Version 'a'
                  [('DT', clf_dt), ('SVM', clf_svm)], # Version 'b'
                  [('DT', clf_dt), ('MLP', clf_mlp)], # Version 'c'
                  [('RF', clf_rf), ('SVM', clf_svm)], # Version 'd'
                  [('RF', clf_rf), ('MLP', clf_mlp)], # Version 'e'
                  [('SVM', clf_svm), ('MLP', clf_mlp)]] # Version 'f'
    index = 0
    version = ['dt_rf', 'dt_svc', 'dt_mlp', 'rf_svc', 'rf_mlp', 'svc_mlp']
    for e in estimators:
        clf_vote2 =  VotingClassifier(estimators = e)
        voting = ['soft']
        weights = [[1,1],[1,2],[1,3],[1,4],[2,1],[2,2],[2,3],[2,4],
                   [3,1],[3,2],[3,3],[3,4],[4,1],[4,2],[4,3],[4,4]]
        n_jobs = [-1]
        params = {'voting' : voting,
                  'weights' : weights,
                  'n_jobs' : n_jobs}
        n_iter = 16
        vote2_random = RandomizedSearchCV(estimator = clf_vote2,
                                          param_distributions = params,
                                          n_iter = n_iter,
                                          scoring = 'roc_auc',
                                          cv = 5,
                                          verbose=2,
                                          random_state=36,
                                          n_jobs = -1)
        search = vote2_random.fit(X_train, y_train)
        scores_vote2 = pd.DataFrame(search.cv_results_)
        scores_vote2.to_csv('scores/vote2_scores_' + version[index] + '.csv')
        title = 'Voting Strategy'
        subs = subtitle + '_2 Classifiers_' + version[index]
        plot_roc(vote2_random.best_estimator_, X_test, y_test, title, subs)
        print(f'Best Parameters for Vote2: {search.best_params_}')
        index +=  1
        
    # Random search for Voting 3, 14 Feats
    estimators = [[('DT', clf_dt), ('MLP', clf_mlp), ('SVM', clf_svm)], # Version 'a'
                  [('RF', clf_rf), ('MLP', clf_mlp), ('SVM', clf_svm)], # Version 'b'
                  [('RF', clf_rf), ('DT', clf_dt), ('SVM', clf_svm)], # Version 'c'
                  [('RF', clf_rf), ('MLP', clf_mlp), ('DT', clf_dt)]] # Version 'd'
    index = 0
    version = ['dt_mlp_svc', 'rf_mlp_svc', 'rf_dt_svc', 'rf_mlp_dt']
    for e in estimators:
        clf_vote3 =  VotingClassifier(estimators = e)
        voting = ['soft']
        weights = [[1,1,1],[1,1,2],[1,1,3],[1,1,4],[1,2,1],[1,2,2],[1,2,3],[1,2,4],
                   [1,3,1],[1,3,2],[1,3,3],[1,3,4],[1,4,1],[1,4,2],[1,4,3],[1,4,4],
                   [2,1,1],[2,1,2],[2,1,3],[2,1,4],[2,2,1],[2,2,2],[2,2,3],[2,2,4],
                   [2,3,1],[2,3,2],[2,3,3],[2,3,4],[2,4,1],[2,4,2],[2,4,3],[2,4,4],
                   [3,1,1],[3,1,2],[3,1,3],[3,1,4],[3,2,1],[3,2,2],[3,2,3],[3,2,4],
                   [3,3,1],[3,3,2],[3,3,3],[3,3,4],[3,4,1],[3,4,2],[3,4,3],[3,4,4],
                   [4,1,1],[4,1,2],[4,1,3],[4,1,4],[4,2,1],[4,2,2],[4,2,3],[4,2,4],
                   [4,3,1],[4,3,2],[4,3,3],[4,3,4],[4,4,1],[4,4,2],[4,4,3],[4,4,4]]
        n_jobs = [-1]
        params = {'voting' : voting,
                  'weights' : weights,
                  'n_jobs' : n_jobs}
        n_iter = 64
        vote3_random = RandomizedSearchCV(estimator = clf_vote3,
                                          param_distributions = params,
                                          n_iter = n_iter,
                                          scoring = 'roc_auc',
                                          cv = 5,
                                          verbose=2,
                                          random_state=36,
                                          n_jobs = -1)
        search = vote3_random.fit(X_train, y_train)
        scores_vote3 = pd.DataFrame(search.cv_results_)
        scores_vote3.to_csv('scores/vote3_scores_' + version[index] + '.csv')
        title = 'Voting Strategy'
        subs = subtitle + '_3 Classifiers_' + version[index]
        plot_roc(vote3_random.best_estimator_, X_test, y_test, title, subs)
        print(f'Best Parameters for Vote3: {search.best_params_}')
        index +=  1
    
    # Random search for Voting 4, 14 Feats
    clf_vote4 =  VotingClassifier(estimators = [('RF', clf_rf), ('DT', clf_dt), ('MLP', clf_mlp), ('SVM', clf_svm)])
    voting = ['soft']
    weights = [[1,1,1,1],[1,1,1,2],[1,1,1,3],[1,1,1,4],[1,1,2,1],[1,1,2,2],[1,1,2,3],[1,1,2,4],
               [1,1,3,1],[1,1,3,2],[1,1,3,3],[1,1,3,4],[1,1,4,1],[1,1,4,2],[1,1,4,3],[1,1,4,4],
               [1,2,1,1],[1,2,1,2],[1,2,1,3],[1,2,1,4],[1,2,2,1],[1,2,2,2],[1,2,2,3],[1,2,2,4],
               [1,2,3,1],[1,2,3,2],[1,2,3,3],[1,2,3,4],[1,2,4,1],[1,2,4,2],[1,2,4,3],[1,2,4,4],
               [1,3,1,1],[1,3,1,2],[1,3,1,3],[1,3,1,4],[1,3,2,1],[1,3,2,2],[1,3,2,3],[1,3,2,4],
               [1,3,3,1],[1,3,3,2],[1,3,3,3],[1,3,3,4],[1,3,4,1],[1,3,4,2],[1,3,4,3],[1,3,4,4],
               [1,4,1,1],[1,4,1,2],[1,4,1,3],[1,4,1,4],[1,4,2,1],[1,4,2,2],[1,4,2,3],[1,4,2,4],
               [1,4,3,1],[1,4,3,2],[1,4,3,3],[1,4,3,4],[1,4,4,1],[1,4,4,2],[1,4,4,3],[1,4,4,4],
               [2,1,1,1],[2,1,1,2],[2,1,1,3],[2,1,1,4],[2,1,2,1],[2,1,2,2],[2,1,2,3],[2,1,2,4],
               [2,1,3,1],[2,1,3,2],[2,1,3,3],[2,1,3,4],[2,1,4,1],[2,1,4,2],[2,1,4,3],[2,1,4,4],
               [2,2,1,1],[2,2,1,2],[2,2,1,3],[2,2,1,4],[2,2,2,1],[2,2,2,2],[2,2,2,3],[2,2,2,4],
               [2,2,3,1],[2,2,3,2],[2,2,3,3],[2,2,3,4],[2,2,4,1],[2,2,4,2],[2,2,4,3],[2,2,4,4],
               [2,3,1,1],[2,3,1,2],[2,3,1,3],[2,3,1,4],[2,3,2,1],[2,3,2,2],[2,3,2,3],[2,3,2,4],
               [2,3,3,1],[2,3,3,2],[2,3,3,3],[2,3,3,4],[2,3,4,1],[2,3,4,2],[2,3,4,3],[2,3,4,4],
               [2,4,1,1],[2,4,1,2],[2,4,1,3],[2,4,1,4],[2,4,2,1],[2,4,2,2],[2,4,2,3],[2,4,2,4],
               [2,4,3,1],[2,4,3,2],[2,4,3,3],[2,4,3,4],[2,4,4,1],[2,4,4,2],[2,4,4,3],[2,4,4,4],
               [3,1,1,1],[3,1,1,2],[3,1,1,3],[3,1,1,4],[3,1,2,1],[3,1,2,2],[3,1,2,3],[3,1,2,4],
               [3,1,3,1],[3,1,3,2],[3,1,3,3],[3,1,3,4],[3,1,4,1],[3,1,4,2],[3,1,4,3],[3,1,4,4],
               [3,2,1,1],[3,2,1,2],[3,2,1,3],[3,2,1,4],[3,2,2,1],[3,2,2,2],[3,2,2,3],[3,2,2,4],
               [3,2,3,1],[3,2,3,2],[3,2,3,3],[3,2,3,4],[3,2,4,1],[3,2,4,2],[3,2,4,3],[3,2,4,4],
               [3,3,1,1],[3,3,1,2],[3,3,1,3],[3,3,1,4],[3,3,2,1],[3,3,2,2],[3,3,2,3],[3,3,2,4],
               [3,3,3,1],[3,3,3,2],[3,3,3,3],[3,3,3,4],[3,3,4,1],[3,3,4,2],[3,3,4,3],[3,3,4,4],
               [3,4,1,1],[3,4,1,2],[3,4,1,3],[3,4,1,4],[3,4,2,1],[3,4,2,2],[3,4,2,3],[3,4,2,4],
               [3,4,3,1],[3,4,3,2],[3,4,3,3],[3,4,3,4],[3,4,4,1],[3,4,4,2],[3,4,4,3],[3,4,4,4],
               [4,1,1,1],[4,1,1,2],[4,1,1,3],[4,1,1,4],[4,1,2,1],[4,1,2,2],[4,1,2,3],[4,1,2,4],
               [4,1,3,1],[4,1,3,2],[4,1,3,3],[4,1,3,4],[4,1,4,1],[4,1,4,2],[4,1,4,3],[4,1,4,4],
               [4,2,1,1],[4,2,1,2],[4,2,1,3],[4,2,1,4],[4,2,2,1],[4,2,2,2],[4,2,2,3],[4,2,2,4],
               [4,2,3,1],[4,2,3,2],[4,2,3,3],[4,2,3,4],[4,2,4,1],[4,2,4,2],[4,2,4,3],[4,2,4,4],
               [4,3,1,1],[4,3,1,2],[4,3,1,3],[4,3,1,4],[4,3,2,1],[4,3,2,2],[4,3,2,3],[4,3,2,4],
               [4,3,3,1],[4,3,3,2],[4,3,3,3],[4,3,3,4],[4,3,4,1],[4,3,4,2],[4,3,4,3],[4,3,4,4],
               [4,4,1,1],[4,4,1,2],[4,4,1,3],[4,4,1,4],[4,4,2,1],[4,4,2,2],[4,4,2,3],[4,4,2,4],
               [4,4,3,1],[4,4,3,2],[4,4,3,3],[4,4,3,4],[4,4,4,1],[4,4,4,2],[4,4,4,3],[4,4,4,4]]
    n_jobs = [-1]
    params = {'voting' : voting,
              'weights' : weights,
              'n_jobs' : n_jobs}
    n_iter = 256
    vote4_random = RandomizedSearchCV(estimator = clf_vote4,
                                   param_distributions = params,
                                   n_iter = n_iter,
                                   scoring = 'roc_auc',
                                   cv = 5,
                                   verbose=2,
                                   random_state=36,
                                   n_jobs = -1)
    search = vote4_random.fit(X_train, y_train)
    scores_vote4 = pd.DataFrame(search.cv_results_)
    scores_vote4.to_csv('scores/vote4_scores' + '.csv')
    title = 'Voting Strategy'
    subs = subtitle + '_4 Classifiers'
    plot_roc(vote4_random.best_estimator_, X_test, y_test, title, subs)
    print(f'Best Parameters for Vote4: {search.best_params_}')

def best_params14():
    best_params = {'dt':{'splitter': 'random', 'min_samples_split': 9, 'min_samples_leaf': 6, 'max_features': 0.3, 'max_depth': 22, 'criterion': 'entropy', 'ccp_alpha': 0.0},
                   'rf':{'n_jobs': -1, 'n_estimators': 431, 'min_samples_split': 10, 'min_samples_leaf': 7, 'max_features': 0.1, 'max_depth': 70, 'criterion': 'gini', 'ccp_alpha': 0.0, 'bootstrap': True},
                   'mlp':{'solver': 'sgd', 'shuffle': True, 'power_t': 0.8, 'learning_rate_init': 0.00894842105263158, 'learning_rate': 'constant', 'hidden_layer_sizes': (15, 200, 80), 'alpha': 0.0005615430861723447, 'activation': 'relu'},
                   'svm':{'shrinking': False, 'probability': True, 'kernel': 'poly', 'gamma': 0.09081818272727273, 'degree': 5, 'coef0': 0.4221105527638191, 'C': 1.282051282051282},
                   'vote2':{'weights': [3, 1], 'voting': 'soft', 'n_jobs': -1}, #nn_rf14
                   'vote3':{'weights': [1, 4, 1], 'voting': 'soft', 'n_jobs': -1}, #dt_nn_svm14
                   'vote4':{'weights': [3, 1, 4, 1], 'voting': 'soft', 'n_jobs': -1}} #dt_nn_svm_rf14
    return best_params

def best_params22():
    best_params = {'dt':{'splitter': 'random', 'min_samples_split': 3, 'min_samples_leaf': 5, 'max_features': 0.6, 'max_depth': 4, 'criterion': 'gini', 'ccp_alpha': 0.0},
                   'rf':{'n_jobs': -1, 'n_estimators': 1265, 'min_samples_split': 7, 'min_samples_leaf': 2, 'max_features': 0.1, 'max_depth': 18, 'criterion': 'gini', 'ccp_alpha': 0.0, 'bootstrap': False},
                   'mlp':{'solver': 'sgd', 'shuffle': False, 'power_t': 0.6, 'learning_rate_init': 0.004216315789473684, 'learning_rate': 'adaptive', 'hidden_layer_sizes': (80, 200, 80), 'alpha': 0.0005397194388777556, 'activation': 'relu'},
                   'svm':{'shrinking': False, 'probability': True, 'kernel': 'rbf', 'gamma': 0.3229090915858586, 'degree': 3, 'coef0': 0.23618090452261306, 'C': 0.3076923076923077},
                   'vote2':{'weights': [4, 2], 'voting': 'soft', 'n_jobs': -1}, #nn_rf22
                   'vote3':{'weights': [4, 4, 1], 'voting': 'soft', 'n_jobs': -1}, #dt_nn_svm22 | Best
                   'vote4':{'weights': [4, 2, 3, 1], 'voting': 'soft', 'n_jobs': -1}} #dt_nn_svm_rf22
    return best_params

def get_best_params():
    best_params = {
        '14': {'dt_nn_svm_rf': {'weights': [3, 1, 4, 1], 'voting': 'soft', 'n_jobs': -1},
              'dt_nn_svm': {'weights': [1, 4, 1], 'voting': 'soft', 'n_jobs': -1},
              'dt_nn': {'weights': [1, 1], 'voting': 'soft', 'n_jobs': -1},
              'dt_rf': {'weights': [1, 4], 'voting': 'soft', 'n_jobs': -1},
              'dt_svm': {'weights': [2, 3], 'voting': 'soft', 'n_jobs': -1},
              'dt': {'splitter': 'random', 'min_samples_split': 9, 'min_samples_leaf': 6, 'max_features': 0.3, 'max_depth': 22, 'criterion': 'entropy', 'ccp_alpha': 0.0},
              'nn_rf_dt': {'weights': [2, 1, 1], 'voting': 'soft', 'n_jobs': -1},
              'nn_rf': {'weights': [3, 1], 'voting': 'soft', 'n_jobs': -1},
              'nn_svm_rf': {'weights': [4, 3, 1], 'voting': 'soft', 'n_jobs': -1},
              'nn_svm':	{'weights': [1, 3], 'voting': 'soft', 'n_jobs': -1},
              'nn': {'solver': 'sgd', 'shuffle': True, 'power_t': 0.8, 'learning_rate_init': 0.00894842105263158, 'learning_rate': 'constant', 'hidden_layer_sizes': (15, 200, 80), 'alpha': 0.0005615430861723447, 'activation': 'relu'},
              'rf': {'n_jobs': -1, 'n_estimators': 431, 'min_samples_split': 10, 'min_samples_leaf': 7, 'max_features': 0.1, 'max_depth': 70, 'criterion': 'gini', 'ccp_alpha': 0.0, 'bootstrap': True},
              'svm_rf_dt': {'weights': [4, 1, 4], 'voting': 'soft', 'n_jobs': -1},
              'svm_rf':	{'weights': [3, 1], 'voting': 'soft', 'n_jobs': -1},
              'svm': {'shrinking': False, 'probability': True, 'kernel': 'poly', 'gamma': 0.09081818272727273, 'degree': 5, 'coef0': 0.4221105527638191, 'C': 1.282051282051282}},
        '22': {'dt_nn_svm_rf': {'weights': [4, 2, 3, 1], 'voting': 'soft', 'n_jobs': -1},
               'dt_nn_svm': {'weights': [4, 4, 1], 'voting': 'soft', 'n_jobs': -1},
               'dt_nn': {'weights': [1, 3], 'voting': 'soft', 'n_jobs': -1},
               'dt_rf':	{'weights': [2, 4], 'voting': 'soft', 'n_jobs': -1},
               'dt_svm': {'weights': [3, 4], 'voting': 'soft', 'n_jobs': -1},
               'dt': {'splitter': 'random', 'min_samples_split': 3, 'min_samples_leaf': 5, 'max_features': 0.6, 'max_depth': 4, 'criterion': 'gini', 'ccp_alpha': 0.0},
               'nn_rf_dt': {'weights': [2, 3, 1], 'voting': 'soft', 'n_jobs': -1},
               'nn_rf': {'weights': [4, 2], 'voting': 'soft', 'n_jobs': -1},
               'nn_svm_rf': {'weights': [1, 2, 1], 'voting': 'soft', 'n_jobs': -1},
               'nn_svm': {'weights': [1, 1], 'voting': 'soft', 'n_jobs': -1},
               'nn': {'solver': 'sgd', 'shuffle': False, 'power_t': 0.6, 'learning_rate_init': 0.004216315789473684, 'learning_rate': 'adaptive', 'hidden_layer_sizes': (80, 200, 80), 'alpha': 0.0005397194388777556, 'activation': 'relu'},
               'rf': {'n_jobs': -1, 'n_estimators': 1265, 'min_samples_split': 7, 'min_samples_leaf': 2, 'max_features': 0.1, 'max_depth': 18, 'criterion': 'gini', 'ccp_alpha': 0.0, 'bootstrap': False},
               'svm_rf_dt':	{'weights': [4, 1, 1], 'voting': 'soft', 'n_jobs': -1},
               'svm_rf': {'weights': [3, 1], 'voting': 'soft', 'n_jobs': -1},
               'svm': {'shrinking': False, 'probability': True, 'kernel': 'rbf', 'gamma': 0.3229090915858586, 'degree': 3, 'coef0': 0.23618090452261306, 'C': 0.3076923076923077}}}
    return best_params


def main():
# =============================================================================
#                        Data Preprocessing
# =============================================================================
    # estate_list = ['BAJA_CALIFORNIA', 'TLAXCALA', 'SONORA', 'COLIMA', 'CDMX']
    estate_list = ['TLAXCALA', 'SONORA']
    ds = preprocess_data(noise_strategy='remove_incomplete',
                         default_value=0,
                         balance_strategy='undersample_mayority',
                         estates = estate_list,
                         extra_symptoms = True)
    X22 = ds.iloc[:, 1:ds.shape[1]-1]
    X14 = X22.iloc[:, [0, 1, 21, 2, 12, 6, 5, 11, 3, 9, 15, 13, 19, 17]] #Paper Englend (14 Contributors Importance Factor)
    # X = X.iloc[:, [11, 9, 1, 0, 5, 19, 8, 3, 2, 15, 6]] #Paper Englend (Clusterization 6 groups)
    # X = X.iloc[:, [1, 21, 2, 12, 6, 5, 11, 3, 9, 15, 13, 19, 17]] #Paper Englend (13 Contributors Importance Factor)
    y = ds.iloc[:, ds.shape[1]-1]
    
    # X22_train, X22_test, y_train, y_test = train_test_split(X22, y, test_size=0.2, random_state=36)
    # X14_train = X22_train.iloc[:, [0, 1, 21, 2, 12, 6, 5, 11, 3, 9, 15, 13, 19, 17]] #Paper Englend (14 Contributors Importance Factor)
    # X14_test = X22_test.iloc[:, [0, 1, 21, 2, 12, 6, 5, 11, 3, 9, 15, 13, 19, 17]] #Paper Englend (14 Contributors Importance Factor)
    
    
    # ds.count()
    # estate_list = ['CDMX']
    # missing_values = check_for_missing_values(estate_list)
    # missing_values.to_csv('report_missing_values.csv')
    
    # Get Best Class 22 feats
    # best_params22Feats = best_params22()
    
    # Get Best Class 14 feats
    # best_params14Feats = best_params14()
    best_params = get_best_params()
    
# =============================================================================
# ############################## Random Search ################################
# =============================================================================
    
    # random_search(X14, y, '14feats')
    # random_search(X22, y, '22feats')
    # random_search_voters(X14, y, '14feats', clf_rf14, clf_dt14, clf_mlp14, clf_svm14)
    # random_search_voters(X22, y, '22feats', clf_rf22, clf_dt22, clf_mlp22, clf_svm22)
    
    
# =============================================================================
# ############################ Baseline 14 feats ##############################
# =============================================================================
    # DT NN SVM RF
    
    clf_dt = tree.DecisionTreeClassifier()
    subtitle = 'decision tree'
    identifier = 'baseline_14'
    results = k_fold_plot_roc(clf_dt, 10, X14, y, subtitle, identifier)
    results_df = pd.DataFrame(data=results)
    
    clf_nn = MLPClassifier()
    subtitle = 'neural network'
    identifier = 'baseline_14'
    results = k_fold_plot_roc(clf_nn, 10, X14, y, subtitle, identifier)
    results_df = pd.concat([results_df, pd.DataFrame(data=results)], axis=0)
    
    clf_svm = svm.SVC(probability=True)
    subtitle = 'support vector machine'
    identifier = 'baseline_14'
    results = k_fold_plot_roc(clf_svm, 10, X14, y, subtitle, identifier)
    results_df = pd.concat([results_df, pd.DataFrame(data=results)], axis=0)
    
    clf_rf = RandomForestClassifier()
    subtitle = 'random forest'
    identifier = 'baseline_14'
    results = k_fold_plot_roc(clf_rf, 10, X14, y, subtitle, identifier)
    results_df = pd.concat([results_df, pd.DataFrame(data=results)], axis=0)
    
    clf_dt_nn = VotingClassifier(estimators = [('DT', clf_dt), ('NN', clf_nn)], voting = 'soft')
    subtitle = 'voting DT & NN'
    identifier = 'baseline_14'
    results = k_fold_plot_roc(clf_dt_nn, 10, X14, y, subtitle, identifier)
    results_df = pd.concat([results_df, pd.DataFrame(data=results)], axis=0)
    
    clf_dt_svm = VotingClassifier(estimators = [('DT', clf_dt), ('SVM', clf_svm)], voting = 'soft')
    subtitle = 'voting DT & SVM'
    identifier = 'baseline_14'
    results = k_fold_plot_roc(clf_dt_svm, 10, X14, y, subtitle, identifier)
    results_df = pd.concat([results_df, pd.DataFrame(data=results)], axis=0)
    
    clf_dt_rf = VotingClassifier(estimators = [('DT', clf_dt), ('RF', clf_rf)], voting = 'soft')
    subtitle = 'voting DT & RF'
    identifier = 'baseline_14'
    results = k_fold_plot_roc(clf_dt_rf, 10, X14, y, subtitle, identifier)
    results_df = pd.concat([results_df, pd.DataFrame(data=results)], axis=0)
    
    clf_nn_svm = VotingClassifier(estimators = [('SVM', clf_svm), ('NN', clf_nn)], voting = 'soft')
    subtitle = 'voting NN & SVM'
    identifier = 'baseline_14'
    results = k_fold_plot_roc(clf_nn_svm, 10, X14, y, subtitle, identifier)
    results_df = pd.concat([results_df, pd.DataFrame(data=results)], axis=0)
    
    clf_nn_rf = VotingClassifier(estimators = [('RF', clf_rf), ('NN', clf_nn)], voting = 'soft')
    subtitle = 'voting NN & RF'
    identifier = 'baseline_14'
    results = k_fold_plot_roc(clf_nn_rf, 10, X14, y, subtitle, identifier)
    results_df = pd.concat([results_df, pd.DataFrame(data=results)], axis=0)
    
    clf_svm_rf = VotingClassifier(estimators = [('RF', clf_rf), ('SVM', clf_svm)], voting = 'soft')
    subtitle = 'voting SVM & RF'
    identifier = 'baseline_14'
    results = k_fold_plot_roc(clf_svm_rf, 10, X14, y, subtitle, identifier)
    results_df = pd.concat([results_df, pd.DataFrame(data=results)], axis=0)
    
    clf_dt_nn_svm = VotingClassifier(estimators = [('DT', clf_dt), ('NN', clf_nn), ('SVM', clf_svm)], voting = 'soft')
    subtitle = 'voting DT, NN & SVM'
    identifier = 'baseline_14'
    results = k_fold_plot_roc(clf_dt_nn_svm, 10, X14, y, subtitle, identifier)
    results_df = pd.concat([results_df, pd.DataFrame(data=results)], axis=0)
    
    clf_dt_nn_rf = VotingClassifier(estimators = [('RF', clf_rf), ('NN', clf_nn), ('DT', clf_dt)], voting = 'soft')
    subtitle = 'voting DT, NN & RF'
    identifier = 'baseline_14'
    results = k_fold_plot_roc(clf_dt_nn_rf, 10, X14, y, subtitle, identifier)
    results_df = pd.concat([results_df, pd.DataFrame(data=results)], axis=0)
    
    clf_dt_svm_rf = VotingClassifier(estimators = [('RF', clf_rf), ('DT', clf_dt), ('SVM', clf_svm)], voting = 'soft')
    subtitle = 'voting DT, SVM & RF'
    identifier = 'baseline_14'
    results = k_fold_plot_roc(clf_dt_svm_rf, 10, X14, y, subtitle, identifier)
    results_df = pd.concat([results_df, pd.DataFrame(data=results)], axis=0)
    
    clf_nn_svm_rf = VotingClassifier(estimators = [('RF', clf_rf), ('NN', clf_nn), ('SVM', clf_svm)], voting = 'soft')
    subtitle = 'voting NN, SVM & RF'
    identifier = 'baseline_14'
    results = k_fold_plot_roc(clf_nn_svm_rf, 10, X14, y, subtitle, identifier)
    results_df = pd.concat([results_df, pd.DataFrame(data=results)], axis=0)
    
    clf_dt_nn_svm_rf = VotingClassifier(estimators = [('RF', clf_rf), ('DT', clf_dt), ('NN', clf_nn), ('SVM', clf_svm)], voting = 'soft')
    subtitle = 'voting DT, NN, SVM & RF'
    identifier = 'baseline_14'
    results = k_fold_plot_roc(clf_dt_nn_svm_rf, 10, X14, y, subtitle, identifier)
    results_df = pd.concat([results_df, pd.DataFrame(data=results)], axis=0)

# =============================================================================
# ############################ Baseline 22 feats ##############################
# =============================================================================
    # DT NN SVM RF
    
    clf_dt = tree.DecisionTreeClassifier()
    subtitle = 'decision tree'
    identifier = 'baseline_22'
    results = k_fold_plot_roc(clf_dt, 10, X22, y, subtitle, identifier)
    results_df = pd.concat([results_df, pd.DataFrame(data=results)], axis=0)
    
    clf_nn = MLPClassifier()
    subtitle = 'neural network'
    identifier = 'baseline_22'
    results = k_fold_plot_roc(clf_nn, 10, X22, y, subtitle, identifier)
    results_df = pd.concat([results_df, pd.DataFrame(data=results)], axis=0)
    
    clf_svm = svm.SVC(probability=True)
    subtitle = 'support vector machine'
    identifier = 'baseline_22'
    results = k_fold_plot_roc(clf_svm, 10, X22, y, subtitle, identifier)
    results_df = pd.concat([results_df, pd.DataFrame(data=results)], axis=0)
    
    clf_rf = RandomForestClassifier()
    subtitle = 'random forest'
    identifier = 'baseline_22'
    results = k_fold_plot_roc(clf_rf, 10, X22, y, subtitle, identifier)
    results_df = pd.concat([results_df, pd.DataFrame(data=results)], axis=0)
    
    clf_dt_nn = VotingClassifier(estimators = [('DT', clf_dt), ('NN', clf_nn)], voting = 'soft')
    subtitle = 'voting DT & NN'
    identifier = 'baseline_22'
    results = k_fold_plot_roc(clf_dt_nn, 10, X22, y, subtitle, identifier)
    results_df = pd.concat([results_df, pd.DataFrame(data=results)], axis=0)
    
    clf_dt_svm = VotingClassifier(estimators = [('DT', clf_dt), ('SVM', clf_svm)], voting = 'soft')
    subtitle = 'voting DT & SVM'
    identifier = 'baseline_22'
    results = k_fold_plot_roc(clf_dt_svm, 10, X22, y, subtitle, identifier)
    results_df = pd.concat([results_df, pd.DataFrame(data=results)], axis=0)
    
    clf_dt_rf = VotingClassifier(estimators = [('DT', clf_dt), ('RF', clf_rf)], voting = 'soft')
    subtitle = 'voting DT & RF'
    identifier = 'baseline_22'
    results = k_fold_plot_roc(clf_dt_rf, 10, X22, y, subtitle, identifier)
    results_df = pd.concat([results_df, pd.DataFrame(data=results)], axis=0)
    
    clf_nn_svm = VotingClassifier(estimators = [('SVM', clf_svm), ('NN', clf_nn)], voting = 'soft')
    subtitle = 'voting NN & SVM'
    identifier = 'baseline_22'
    results = k_fold_plot_roc(clf_nn_svm, 10, X22, y, subtitle, identifier)
    results_df = pd.concat([results_df, pd.DataFrame(data=results)], axis=0)
    
    clf_nn_rf = VotingClassifier(estimators = [('RF', clf_rf), ('NN', clf_nn)], voting = 'soft')
    subtitle = 'voting NN & RF'
    identifier = 'baseline_22'
    results = k_fold_plot_roc(clf_nn_rf, 10, X22, y, subtitle, identifier)
    results_df = pd.concat([results_df, pd.DataFrame(data=results)], axis=0)
    
    clf_svm_rf = VotingClassifier(estimators = [('RF', clf_rf), ('SVM', clf_svm)], voting = 'soft')
    subtitle = 'voting SVM & RF'
    identifier = 'baseline_22'
    results = k_fold_plot_roc(clf_svm_rf, 10, X22, y, subtitle, identifier)
    results_df = pd.concat([results_df, pd.DataFrame(data=results)], axis=0)
    
    clf_dt_nn_svm = VotingClassifier(estimators = [('DT', clf_dt), ('NN', clf_nn), ('SVM', clf_svm)], voting = 'soft')
    subtitle = 'voting DT, NN & SVM'
    identifier = 'baseline_22'
    results = k_fold_plot_roc(clf_dt_nn_svm, 10, X22, y, subtitle, identifier)
    results_df = pd.concat([results_df, pd.DataFrame(data=results)], axis=0)
    
    clf_dt_nn_rf = VotingClassifier(estimators = [('RF', clf_rf), ('NN', clf_nn), ('DT', clf_dt)], voting = 'soft')
    subtitle = 'voting DT, NN & RF'
    identifier = 'baseline_22'
    results = k_fold_plot_roc(clf_dt_nn_rf, 10, X22, y, subtitle, identifier)
    results_df = pd.concat([results_df, pd.DataFrame(data=results)], axis=0)
    
    clf_dt_svm_rf = VotingClassifier(estimators = [('RF', clf_rf), ('DT', clf_dt), ('SVM', clf_svm)], voting = 'soft')
    subtitle = 'voting DT, SVM & RF'
    identifier = 'baseline_22'
    results = k_fold_plot_roc(clf_dt_svm_rf, 10, X22, y, subtitle, identifier)
    results_df = pd.concat([results_df, pd.DataFrame(data=results)], axis=0)
    
    clf_nn_svm_rf = VotingClassifier(estimators = [('RF', clf_rf), ('NN', clf_nn), ('SVM', clf_svm)], voting = 'soft')
    subtitle = 'voting NN, SVM & RF'
    identifier = 'baseline_22'
    results = k_fold_plot_roc(clf_nn_svm_rf, 10, X22, y, subtitle, identifier)
    results_df = pd.concat([results_df, pd.DataFrame(data=results)], axis=0)
    
    clf_dt_nn_svm_rf = VotingClassifier(estimators = [('RF', clf_rf), ('DT', clf_dt), ('NN', clf_nn), ('SVM', clf_svm)], voting = 'soft')
    subtitle = 'voting DT, NN, SVM & RF'
    identifier = 'baseline_22'
    results = k_fold_plot_roc(clf_dt_nn_svm_rf, 10, X22, y, subtitle, identifier)
    results_df = pd.concat([results_df, pd.DataFrame(data=results)], axis=0)
    

# =============================================================================
# ############################# Tunned 14 feats ###############################
# =============================================================================
    # DT NN SVM RF
    
    clf_dt = tree.DecisionTreeClassifier(**best_params['14']['dt'])
    subtitle = 'decision tree'
    identifier = 'tunned_14'
    results = k_fold_plot_roc(clf_dt, 10, X14, y, subtitle, identifier)
    results_df = pd.concat([results_df, pd.DataFrame(data=results)], axis=0)
    
    clf_nn = MLPClassifier(**best_params['14']['nn'])
    subtitle = 'neural network'
    identifier = 'tunned_14'
    results = k_fold_plot_roc(clf_nn, 10, X14, y, subtitle, identifier)
    results_df = pd.concat([results_df, pd.DataFrame(data=results)], axis=0)
    
    clf_svm = svm.SVC(**best_params['14']['svm'])
    subtitle = 'support vector machine'
    identifier = 'tunned_14'
    results = k_fold_plot_roc(clf_svm, 10, X14, y, subtitle, identifier)
    results_df = pd.concat([results_df, pd.DataFrame(data=results)], axis=0)
    
    clf_rf = RandomForestClassifier(**best_params['14']['rf'])
    subtitle = 'random forest'
    identifier = 'tunned_14'
    results = k_fold_plot_roc(clf_rf, 10, X14, y, subtitle, identifier)
    results_df = pd.concat([results_df, pd.DataFrame(data=results)], axis=0)
    
    clf_dt_nn = VotingClassifier(estimators = [('DT', clf_dt), ('NN', clf_nn)], **best_params['14']['dt_nn'])
    subtitle = 'voting DT & NN'
    identifier = 'tunned_14'
    results = k_fold_plot_roc(clf_dt_nn, 10, X14, y, subtitle, identifier)
    results_df = pd.concat([results_df, pd.DataFrame(data=results)], axis=0)
    
    clf_dt_svm = VotingClassifier(estimators = [('DT', clf_dt), ('SVM', clf_svm)], **best_params['14']['dt_svm'])
    subtitle = 'voting DT & SVM'
    identifier = 'tunned_14'
    results = k_fold_plot_roc(clf_dt_svm, 10, X14, y, subtitle, identifier)
    results_df = pd.concat([results_df, pd.DataFrame(data=results)], axis=0)
    
    clf_dt_rf = VotingClassifier(estimators = [('DT', clf_dt), ('RF', clf_rf)], **best_params['14']['dt_rf'])
    subtitle = 'voting DT & RF'
    identifier = 'tunned_14'
    results = k_fold_plot_roc(clf_dt_rf, 10, X14, y, subtitle, identifier)
    results_df = pd.concat([results_df, pd.DataFrame(data=results)], axis=0)
    
    clf_nn_svm = VotingClassifier(estimators = [('SVM', clf_svm), ('NN', clf_nn)], **best_params['14']['nn_svm'])
    subtitle = 'voting NN & SVM'
    identifier = 'tunned_14'
    results = k_fold_plot_roc(clf_nn_svm, 10, X14, y, subtitle, identifier)
    results_df = pd.concat([results_df, pd.DataFrame(data=results)], axis=0)
    
    clf_nn_rf = VotingClassifier(estimators = [('RF', clf_rf), ('NN', clf_nn)], **best_params['14']['nn_rf'])
    subtitle = 'voting NN & RF'
    identifier = 'tunned_14'
    results = k_fold_plot_roc(clf_nn_rf, 10, X14, y, subtitle, identifier)
    results_df = pd.concat([results_df, pd.DataFrame(data=results)], axis=0)
    
    clf_svm_rf = VotingClassifier(estimators = [('RF', clf_rf), ('SVM', clf_svm)], **best_params['14']['svm_rf'])
    subtitle = 'voting SVM & RF'
    identifier = 'tunned_14'
    results = k_fold_plot_roc(clf_svm_rf, 10, X14, y, subtitle, identifier)
    results_df = pd.concat([results_df, pd.DataFrame(data=results)], axis=0)
    
    clf_dt_nn_svm = VotingClassifier(estimators = [('DT', clf_dt), ('NN', clf_nn), ('SVM', clf_svm)], **best_params['14']['dt_nn_svm'])
    subtitle = 'voting DT, NN & SVM'
    identifier = 'tunned_14'
    results = k_fold_plot_roc(clf_dt_nn_svm, 10, X14, y, subtitle, identifier)
    results_df = pd.concat([results_df, pd.DataFrame(data=results)], axis=0)
    
    clf_dt_nn_rf = VotingClassifier(estimators = [('RF', clf_rf), ('NN', clf_nn), ('DT', clf_dt)], **best_params['14']['nn_rf_dt'])
    subtitle = 'voting DT, NN & RF'
    identifier = 'tunned_14'
    results = k_fold_plot_roc(clf_dt_nn_rf, 10, X14, y, subtitle, identifier)
    results_df = pd.concat([results_df, pd.DataFrame(data=results)], axis=0)
    
    clf_dt_svm_rf = VotingClassifier(estimators = [('RF', clf_rf), ('DT', clf_dt), ('SVM', clf_svm)], **best_params['14']['svm_rf_dt'])
    subtitle = 'voting DT, SVM & RF'
    identifier = 'tunned_14'
    results = k_fold_plot_roc(clf_dt_svm_rf, 10, X14, y, subtitle, identifier)
    results_df = pd.concat([results_df, pd.DataFrame(data=results)], axis=0)
    
    clf_nn_svm_rf = VotingClassifier(estimators = [('RF', clf_rf), ('NN', clf_nn), ('SVM', clf_svm)], **best_params['14']['nn_svm_rf'])
    subtitle = 'voting NN, SVM & RF'
    identifier = 'tunned_14'
    results = k_fold_plot_roc(clf_nn_svm_rf, 10, X14, y, subtitle, identifier)
    results_df = pd.concat([results_df, pd.DataFrame(data=results)], axis=0)
    
    clf_dt_nn_svm_rf = VotingClassifier(estimators = [('RF', clf_rf), ('DT', clf_dt), ('NN', clf_nn), ('SVM', clf_svm)], **best_params['14']['dt_nn_svm_rf'])
    subtitle = 'voting DT, NN, SVM & RF'
    identifier = 'tunned_14'
    results = k_fold_plot_roc(clf_dt_nn_svm_rf, 10, X14, y, subtitle, identifier)
    results_df = pd.concat([results_df, pd.DataFrame(data=results)], axis=0)

# =============================================================================
# ############################# Tunned 22 feats ###############################
# =============================================================================
    # DT NN SVM RF
    
    clf_dt = tree.DecisionTreeClassifier(**best_params['22']['dt'])
    subtitle = 'decision tree'
    identifier = 'tunned_22'
    results = k_fold_plot_roc(clf_dt, 10, X22, y, subtitle, identifier)
    results_df = pd.concat([results_df, pd.DataFrame(data=results)], axis=0)
    
    clf_nn = MLPClassifier(**best_params['22']['nn'])
    subtitle = 'neural network'
    identifier = 'tunned_22'
    results = k_fold_plot_roc(clf_nn, 10, X22, y, subtitle, identifier)
    results_df = pd.concat([results_df, pd.DataFrame(data=results)], axis=0)
    
    clf_svm = svm.SVC(**best_params['22']['svm'])
    subtitle = 'support vector machine'
    identifier = 'tunned_22'
    results = k_fold_plot_roc(clf_svm, 10, X22, y, subtitle, identifier)
    results_df = pd.concat([results_df, pd.DataFrame(data=results)], axis=0)
    
    clf_rf = RandomForestClassifier(**best_params['22']['rf'])
    subtitle = 'random forest'
    identifier = 'tunned_22'
    results = k_fold_plot_roc(clf_rf, 10, X22, y, subtitle, identifier)
    results_df = pd.concat([results_df, pd.DataFrame(data=results)], axis=0)
    
    clf_dt_nn = VotingClassifier(estimators = [('DT', clf_dt), ('NN', clf_nn)], **best_params['22']['dt_nn'])
    subtitle = 'voting DT & NN'
    identifier = 'tunned_22'
    results = k_fold_plot_roc(clf_dt_nn, 10, X22, y, subtitle, identifier)
    results_df = pd.concat([results_df, pd.DataFrame(data=results)], axis=0)
    
    clf_dt_svm = VotingClassifier(estimators = [('DT', clf_dt), ('SVM', clf_svm)], **best_params['22']['dt_svm'])
    subtitle = 'voting DT & SVM'
    identifier = 'tunned_22'
    results = k_fold_plot_roc(clf_dt_svm, 10, X22, y, subtitle, identifier)
    results_df = pd.concat([results_df, pd.DataFrame(data=results)], axis=0)
    
    clf_dt_rf = VotingClassifier(estimators = [('DT', clf_dt), ('RF', clf_rf)], **best_params['22']['dt_rf'])
    subtitle = 'voting DT & RF'
    identifier = 'tunned_22'
    results = k_fold_plot_roc(clf_dt_rf, 10, X22, y, subtitle, identifier)
    results_df = pd.concat([results_df, pd.DataFrame(data=results)], axis=0)
    
    clf_nn_svm = VotingClassifier(estimators = [('SVM', clf_svm), ('NN', clf_nn)], **best_params['22']['nn_svm'])
    subtitle = 'voting NN & SVM'
    identifier = 'tunned_22'
    results = k_fold_plot_roc(clf_nn_svm, 10, X22, y, subtitle, identifier)
    results_df = pd.concat([results_df, pd.DataFrame(data=results)], axis=0)
    
    clf_nn_rf = VotingClassifier(estimators = [('RF', clf_rf), ('NN', clf_nn)], **best_params['22']['nn_rf'])
    subtitle = 'voting NN & RF'
    identifier = 'tunned_22'
    results = k_fold_plot_roc(clf_nn_rf, 10, X22, y, subtitle, identifier)
    results_df = pd.concat([results_df, pd.DataFrame(data=results)], axis=0)
    
    clf_svm_rf = VotingClassifier(estimators = [('RF', clf_rf), ('SVM', clf_svm)], **best_params['22']['svm_rf'])
    subtitle = 'voting SVM & RF'
    identifier = 'tunned_22'
    results = k_fold_plot_roc(clf_svm_rf, 10, X22, y, subtitle, identifier)
    results_df = pd.concat([results_df, pd.DataFrame(data=results)], axis=0)
    
    clf_dt_nn_svm = VotingClassifier(estimators = [('DT', clf_dt), ('NN', clf_nn), ('SVM', clf_svm)], **best_params['22']['dt_nn_svm'])
    subtitle = 'voting DT, NN & SVM'
    identifier = 'tunned_22'
    results = k_fold_plot_roc(clf_dt_nn_svm, 10, X22, y, subtitle, identifier)
    results_df = pd.concat([results_df, pd.DataFrame(data=results)], axis=0)
    
    clf_dt_nn_rf = VotingClassifier(estimators = [('RF', clf_rf), ('NN', clf_nn), ('DT', clf_dt)], **best_params['22']['nn_rf_dt'])
    subtitle = 'voting DT, NN & RF'
    identifier = 'tunned_22'
    results = k_fold_plot_roc(clf_dt_nn_rf, 10, X22, y, subtitle, identifier)
    results_df = pd.concat([results_df, pd.DataFrame(data=results)], axis=0)
    
    clf_dt_svm_rf = VotingClassifier(estimators = [('RF', clf_rf), ('DT', clf_dt), ('SVM', clf_svm)], **best_params['22']['svm_rf_dt'])
    subtitle = 'voting DT, SVM & RF'
    identifier = 'tunned_22'
    results = k_fold_plot_roc(clf_dt_svm_rf, 10, X22, y, subtitle, identifier)
    results_df = pd.concat([results_df, pd.DataFrame(data=results)], axis=0)
    
    clf_nn_svm_rf = VotingClassifier(estimators = [('RF', clf_rf), ('NN', clf_nn), ('SVM', clf_svm)], **best_params['22']['nn_svm_rf'])
    subtitle = 'voting NN, SVM & RF'
    identifier = 'tunned_22'
    results = k_fold_plot_roc(clf_nn_svm_rf, 10, X22, y, subtitle, identifier)
    results_df = pd.concat([results_df, pd.DataFrame(data=results)], axis=0)
    
    clf_dt_nn_svm_rf = VotingClassifier(estimators = [('RF', clf_rf), ('DT', clf_dt), ('NN', clf_nn), ('SVM', clf_svm)], **best_params['22']['dt_nn_svm_rf'])
    subtitle = 'voting DT, NN, SVM & RF'
    identifier = 'tunned_22'
    results = k_fold_plot_roc(clf_dt_nn_svm_rf, 10, X22, y, subtitle, identifier)
    results_df = pd.concat([results_df, pd.DataFrame(data=results)], axis=0).reset_index(drop=True)
    
# =============================================================================
# ############################ Save results to CSV ############################
# =============================================================================
    results_df.to_csv('final_results.csv')
    
# =============================================================================
# ############################## Best Classifier ##############################
# classifier: voting DT & RF_tunned_22_10fold,
# sensitivity: 0.741585802,
# specificity: 0.601391726,
# precision: 0.651793728,
# f1: 0.6614954,
# accuracy: 0.671538755,
# mean_auc: 0.729834989,
# std_auc: 0.044231013
# =============================================================================

    clf_best = VotingClassifier(estimators = [('DT', clf_dt), ('RF', clf_rf)], **best_params['22']['dt_rf'])
    subtitle = 'voting DT & RF'
    identifier = 'best_22'
    results = k_fold_plot_roc(clf_best, 10, X22, y, subtitle, identifier)
    
    clf_best = VotingClassifier(estimators = [('DT', clf_dt), ('RF', clf_rf)], **best_params['22']['dt_rf'])
    subtitle = 'voting DT & RF'
    identifier = 'best_14'
    results = k_fold_plot_roc(clf_best, 10, X14, y, subtitle, identifier)
    
    
    #Test Best vs CDMX
    estate_list = ['CDMX']
    ds_full = preprocess_data(noise_strategy='remove_incomplete',
                         default_value=0,
                         balance_strategy='undersample_mayority',
                         estates = estate_list,
                         extra_symptoms = False)
    Xf = ds_full.iloc[:, 1:ds_full.shape[1]-1]
    Xf.insert(19,'DYSGEUSIA',Xf['KNOWN_COVID_CONTACT']) #.iloc[:, 21] = Xf.iloc[:, 19]
    Xf.insert(19,'ANOSMIA',Xf['KNOWN_COVID_CONTACT']) #.iloc[:, 21] = Xf.iloc[:, 19]
    Xf.loc[:, 'DYSGEUSIA'] = 0
    Xf.loc[:, 'ANOSMIA'] = 0
    yf = ds_full.iloc[:, ds_full.shape[1]-1]
    clf_best = VotingClassifier(estimators = [('DT', clf_dt), ('RF', clf_rf)], **best_params['22']['dt_rf'])
    clf_best = clf_best.fit(X22, y)
    title = 'best classifier'
    subtitle = 'cdmx'
    plot_roc(clf_best, Xf, yf, title, subtitle)
    
    # clf = eval_model(clf, X, y, 5)
    # clf = clf.fit(X, y)
    # joblib.dump(clf, 'app/covid19_model.pkl')
    
    print('Success!')
    return ds

def naive_bayes():
    #Bernoulli Naive Bayes
    estate_list = ['TLAXCALA', 'SONORA']
    ds = preprocess_data(noise_strategy='remove_incomplete',
                         default_value=0,
                         balance_strategy='None',
                         estates = estate_list,
                         extra_symptoms = True)
    X = ds.iloc[:, 1:ds.shape[1]-1]
    y = ds.iloc[:, ds.shape[1]-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=36)
    clf_bnb = BernoulliNB()
    clf_bnb = clf_bnb.fit(X_train, y_train)
    title = 'Bernoulli Naive Bayes (14 Contributors)'
    plot_roc(clf_bnb, X_test, y_test, title)
    
    
    #MultinomialNB
    estate_list = ['TLAXCALA', 'SONORA']
    ds = preprocess_data(noise_strategy='remove_incomplete',
                         default_value=0,
                         balance_strategy='undersample_mayority',
                         estates = estate_list,
                         extra_symptoms = True)
    X = ds.iloc[:, 1:ds.shape[1]-1]
    y = ds.iloc[:, ds.shape[1]-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=36)
    clf_mnb = MultinomialNB()
    clf_mnb = clf_mnb.fit(X_train, y_train)
    title = 'Multinomial Naive Bayes (14 Contributors)'
    plot_roc(clf_mnb, X_test, y_test, title)
    
    
    #Gaussian Naive Bayes
    estate_list = ['TLAXCALA', 'SONORA']
    ds = preprocess_data(noise_strategy='remove_incomplete',
                         default_value=0,
                         balance_strategy='undersample_mayority',
                         estates = estate_list,
                         extra_symptoms = True)
    X = ds.iloc[:, 1:ds.shape[1]-1]
    y = ds.iloc[:, ds.shape[1]-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=36)
    clf_gnb = GaussianNB()
    clf_gnb = clf_gnb.fit(X_train, y_train)
    title = 'Gaussian Naive Bayes (14 Contributors)'
    plot_roc(clf_gnb, X_test, y_test, title)

if __name__ == '__main__':
    ds = main()
