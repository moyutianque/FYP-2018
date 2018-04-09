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
    pos_csv = '.\\test_dataset\\test_pos.csv'
    neg_csv = '.\\test_dataset\\test_neg.csv'
    m_path = '.\model\learned_model48.pkl'
    df1 = pd.read_csv(pos_csv)
    df2 = pd.read_csv(neg_csv)
    
    # eliminate NaN row
    #df1 = df1.dropna(axis=0, how='any')
    #df2 = df2.dropna(axis=0, how='any')
    
    #print(df2.info)
    pos = df1.values
    neg = df2.values
    
    X = np.r_[pos,neg] # c_ and r_ is used to concatenate array
    Y = [1] * pos.shape[0] + [0] * neg.shape[0]
    
    if os.path.exists(m_path):
        clf = joblib.load(m_path) 
    else:
        print("[INFO] model not exist")
        
    print("[INFO] running test")
    score = clf.score(X,Y)
    print("Accuracy: %0.2f" % score)
    

if __name__ == '__main__':
    main()

    