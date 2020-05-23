#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 01:48:18 2020

@author: roberto
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier


def preprocess_symptoms_only():
    rawDs = pd.read_csv("covid19_tlax_data_raw.csv").sample(frac=1).sample(frac=1).sample(frac=1)

    #Dataset solo sintomas
    ds = pd.concat([rawDs.iloc[:, 31:49], rawDs.iloc[:, 76]], axis=1)
    ds = pd.concat([ds.loc[ds['RESULTADO DEFINITIVO'] == 'NEGATIVO', :], ds.loc[ds['RESULTADO DEFINITIVO'] == 'SARS-CoV-2', :]], axis=0).sample(frac=1).reset_index(drop=True)

    ds = ds.replace('NO', 0)
    ds = ds.replace('SI', 1)
    ds = ds.replace('NEGATIVO', 0)
    ds = ds.replace('SARS-CoV-2', 1)

    return ds

def preprocess_symptoms_contact_sudden():
    rawDs = pd.read_csv("covid19_tlax_data_raw.csv").sample(frac=1).sample(frac=1).sample(frac=1)

    #Dtaset sintomas, contacto y subito
    ds = pd.concat([rawDs.iloc[:, 31:50], rawDs.iloc[:, [65, 76]]], axis=1)
    ds = pd.concat([ds.loc[ds['RESULTADO DEFINITIVO'] == 'NEGATIVO', :], ds.loc[ds['RESULTADO DEFINITIVO'] == 'SARS-CoV-2', :]], axis=0).sample(frac=1).reset_index(drop=True)

    ds = ds.replace('NO', 0)
    ds = ds.replace('SI', 1)
    ds = ds.replace('NEGATIVO', 0)
    ds = ds.replace('SARS-CoV-2', 1)

    return ds

def preprocess_symptoms_full():
    rawDs = pd.read_csv("covid19_tlax_data_raw.csv").sample(frac=1).sample(frac=1).sample(frac=1)
    auxDs = rawDs.iloc[:, [4, 5, 20, 22, 23, 29]]

    columnFReg = auxDs.iloc[:,0].apply(lambda x: datetime.strptime(x, '%d/%m/%Y'))
    columnFInicioSint = auxDs.iloc[:,5].apply(lambda x: datetime.strptime(x, '%d/%m/%Y'))
    columnFNac = auxDs.iloc[:,2].apply(lambda x: datetime.strptime(x, '%d/%m/%Y'))

    columnSymptDays = (columnFReg - columnFInicioSint)/np.timedelta64(1,'D')
    columnSymptDays.name = "DIAS_INICIO_SINTOMAS"

    columnAgesYrs = (columnFReg - columnFNac)/np.timedelta64(1,'Y')
    columnAgesYrs.name = "EDAD_ANIOS"

    columnsSexo = pd.get_dummies(auxDs['SEXO'], prefix='SEXO')

    scaledSymptAge = pd.concat([columnAgesYrs, columnSymptDays], axis=1)

    scaler = MinMaxScaler()
    scaledSymptAge[['DIAS_INICIO_SINTOMAS', 'EDAD_ANIOS']] = scaler.fit_transform(scaledSymptAge[['DIAS_INICIO_SINTOMAS', 'EDAD_ANIOS']])

    #Dtaset sintomas y otros
    ds = pd.concat([columnsSexo, scaledSymptAge, auxDs.iloc[:, [3, 4]], rawDs.iloc[:, 31:50], rawDs.iloc[:, [65, 76]]], axis=1)
    ds = pd.concat([ds.loc[ds['RESULTADO DEFINITIVO'] == 'NEGATIVO', :], ds.loc[ds['RESULTADO DEFINITIVO'] == 'SARS-CoV-2', :]], axis=0).sample(frac=1).reset_index(drop=True)

    ds = ds.replace('NO', 0)
    ds = ds.replace('SI', 1)
    ds = ds.replace('NEGATIVO', 0)
    ds = ds.replace('SARS-CoV-2', 1)

    return ds


def train_rf(X, y):
    X, Xt, y, yt = train_test_split(X, y, test_size=0.2, random_state = 0)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    Xt = Xt.reset_index(drop=True)
    yt = yt.reset_index(drop=True)

    clf = RandomForestClassifier(n_estimators=1000)
    clf = clf.fit(X, y)
    ypred = clf.predict(Xt)
    return clf, confusion_matrix(yt, ypred)

def main():
    ds = pd.DataFrame(preprocess_symptoms_contact_sudden())
    
    #NaN y "SE IGNORA" se ponen a AVG
    for column in ds:
        if (column != 'RESULTADO DEFINITIVO'):
            colSUM = ds.loc[ds[column] == 1, column].sum()
            colAVG = colSUM/ds.shape[0]
            ds[column] = ds[column].fillna(colAVG)
            ds[column] = ds[column].replace('SE IGNORA', colAVG)

    #Sintomas, Contacto y Súbito
    X = ds.iloc[:, :20]
    y = ds.iloc[:, 20]

    clf, cm = train_rf(X, y)
    file = open('app/rft1000.pkl', 'wb')
    pickle.dump(clf, file)
    file.close()
    print('Success!')
    return ds


if __name__ == "__main__":
    ds = main()
