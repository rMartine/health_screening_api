# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 15:10:38 2020

@author: rober
"""

import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


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

def consolidate_scores(best_scores, run_scores):
    df = pd.DataFrame(columns=['feats', 'mean_roc', 'std_roc'])
    df = pd.concat([best_scores, run_scores], axis=0)
    df = df.sort_values(by=['mean_roc'], ascending=False).reset_index(drop=True)
    df = df.iloc[0:3000, :]
    return df

def feature_selector(features, X, y):
    # std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=36)
    run_scores = pd.DataFrame(columns=['feats', 'mean_roc', 'std_roc'])
    feature_list = []
    for f in features:
        # print('%d. feature %d (%f)' % (f + 1, indices[f], importances[indices[f]]))
        feature_list.append(f)
        X_new = X.iloc[:, feature_list]
        mean_roc_auc, std_roc_auc = mean_roc(X_new, y, 10)
        # print ('MEAN_ROC_AUC=%f, STD_ROC_AUC=%f,TOP=%d' % (mean_roc_auc, std_roc_auc, f+1))
        aux_df = pd.DataFrame({'feats': str(feature_list).strip('[]'), 'mean_roc': mean_roc_auc, 'std_roc': std_roc_auc}, index=[0])
        run_scores = pd.concat([run_scores, aux_df], axis = 0)
    return run_scores

def mean_roc(X, y, folds):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    forest = RandomForestClassifier(max_depth=4)
    cv = StratifiedKFold(n_splits=folds)
    cv.get_n_splits(X, y)
    for i, (train, test) in enumerate(cv.split(X, y)):
        forest.fit(X.iloc[train], y.iloc[train])
        probs = forest.predict_proba(X.iloc[test])
        preds = probs[:,1]
        fpr, tpr, threshold = roc_curve(y.iloc[test], preds)
        roc_auc = auc(fpr, tpr)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_auc)
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    return mean_auc, std_auc

def permutator(symptoms, P, X, y):
    best_scores = pd.DataFrame(columns=['feats', 'mean_roc', 'std_roc'])
    if (len(symptoms) > 1):
        for s in symptoms:
            permutation = P + [s]
            filteredsymptoms = [symptom for symptom in symptoms if symptom != s]
            run_scores = permutator(filteredsymptoms, permutation, X, y)
            best_scores = consolidate_scores(best_scores, run_scores)
    else:
        P.append(symptoms[0])
        print(P)
        return feature_selector(P, X, y)
    return best_scores

def combinator(symptoms, X, y):
    combination_list = []
    best_scores = pd.DataFrame(columns=['feats', 'mean_roc', 'std_roc'])
    # symptoms = [int(x) for x in range(22)]
    for i in range(len(symptoms)):
        combination_list = combination_list + list(combinations(symptoms, i+1))
    for s in combination_list:
        print(s)
        # run_scores = feature_selector(s, X, y)
        X_new = X.iloc[:, list(s)]
        mean_roc_auc, std_roc_auc = mean_roc(X_new, y, 6)
        # print ('MEAN_ROC_AUC=%f, STD_ROC_AUC=%f,TOP=%d' % (mean_roc_auc, std_roc_auc, f+1))
        aux_df = pd.DataFrame({'feats': str(list(s)).strip('[]'), 'mean_roc': mean_roc_auc, 'std_roc': std_roc_auc}, index=[0])
        best_scores = consolidate_scores(best_scores, aux_df)
    return best_scores

def main():
    estate_list = ['TLAXCALA', 'SONORA']
    ds = preprocess_data(noise_strategy='remove_incomplete',
                         default_value=0,
                         balance_strategy='undersample_mayority',
                         estates = estate_list,
                         extra_symptoms = True)
    X = ds.iloc[:, 1:ds.shape[1]-1]
    y = ds.iloc[:, ds.shape[1]-1]
    
    # symptoms = np.array(['A', 'B', 'C', 'D'])
    symptoms = [int(x) for x in range(22)]
    # best_scores = permutator(symptoms, [], X, y)
    best_scores = combinator(symptoms, X, y)
    best_scores.to_csv('top3000_clusters.csv')
    # ds.count()
    
    print('Success!')
    return ds

if __name__ == '__main__':
    ds = main()