#!/usr/bin/python  
# -*- coding: utf-8 -*- 
import os
import dlib
import cv2
import time
import numpy as np
from imutils import face_utils
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score

from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

def main():
    time1 = time.localtime(time.time())
    l_data = 'train_pos.csv'
    il_data = 'train_neg.csv'
    m_path = '.\model\learned_model%s.pkl'%(str(time1.tm_mon)+str(time1.tm_mday))
    df1 = pd.read_csv(l_data)
    df2 = pd.read_csv(il_data)
    
    # eliminate NaN row
    df1 = df1.dropna(axis=0, how='any')
    df2 = df2.dropna(axis=0, how='any')
    
    #print(df2.info)
    pos = df1.values
    neg = df2.values
    
    X = np.r_[pos,neg] # c_ and r_ is used to concatenate array
    Y = [1] * pos.shape[0] + [0] * neg.shape[0]
    

    print("[INFO] start training") 
    num_e = 5
    score = 0
    
    for i in range(50,60): # index tuning
        print("Round: %d" % (i))
        clf = RandomForestClassifier(n_estimators=i,min_samples_split=2)
        #clf = AdaBoostClassifier(n_estimators=i)
        #clf = GradientBoostingClassifier(n_estimators=i)
        scores = cross_val_score(clf, X, Y, cv=10, scoring='accuracy')
        print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(),'Random Forest'))
        
        if scores.mean() > score:
            score = scores.mean()
            num_e = i
    
    #clf = RandomForestClassifier(n_estimators=num_e,min_samples_split=2) 
    #clf = svm.SVC()
    #clf = AdaBoostClassifier(n_estimators=num_e)      
    #clf = GradientBoostingClassifier(n_estimators=num_e)
    clf.fit(X,Y)    
    joblib.dump(clf, m_path)
    print(num_e)

if __name__ == '__main__':
    main()

    