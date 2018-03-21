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


def main():
    time1 = time.localtime(time.time())
    l_data = 'laugh.csv'
    il_data = 'instead_laugh.csv'
    m_path = '.\model\learned_model%s.pkl'%(str(time1.tm_mon)+str(time1.tm_mday))
    df1 = pd.read_csv(l_data)
    df2 = pd.read_csv(il_data)
    #print(df2.info)
    pos = df1.values
    neg = df2.values
    
    X = np.r_[pos,neg] # c_ and r_ is used to concatenate array
    Y = [1] * pos.shape[0] + [0] * neg.shape[0]
    
    if os.path.exists(m_path):
        clf = joblib.load(m_path) 
    else:
        print("[INFO] start training")
        clf = RandomForestClassifier(n_estimators=100,min_samples_split=2)
        clf.fit(X,Y)
        joblib.dump(clf, m_path)
        
    # test1
    print("[INFO] running test1")
    scores = cross_val_score(clf, X, Y, cv=5, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(),'Random Forest'))
    
    #test2 webcam
    print("[INFO] running test2")
    x_test = np.asarray([[1.838462802,1.850380919,1.813466246,1.586179729,0.77206562,0.826866573,2.618314331,1.059522838,1.142230869,1.169812414]])
    print(clf.predict(x_test))
    
if __name__ == '__main__':
    main()

    