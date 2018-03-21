#!/usr/bin/python  
# -*- coding: utf-8 -*- 
import numpy as np
from scipy.spatial import distance as dist

def cal_angle(x1, y1, x2, y2):
    cos_angle = (x1*x2+y1*y2)/(np.sqrt(np.power(x1,2)+np.power(y1,2))*
                               np.sqrt(np.power(x2,2)+np.power(y2,2)))
    angle = np.arccos(cos_angle)
    while (angle<0):
        angle += np.pi
    while (angle>np.pi):
        angle -= np.pi
    #angle = angle*360/2/np.pi
    return angle

def angle_based_on_index(shape,idx1,idx2,idx3):
    [x1,y1] = shape[idx1-1]-shape[idx2-1]
    [x2,y2] = shape[idx3-1]-shape[idx2-1]
    return cal_angle(x1,y1,x2,y2)
    
def extract_features(shape):         
    feature_list = []
    # feature 1 angle no.49,no.58,no.55
    feature_list.extend([angle_based_on_index(shape,49,58,55)])
    
    # feature 2 angle no.49,no.59,no.55
    feature_list.extend([angle_based_on_index(shape,49,59,55)])
    
    # feature 3 angle no.49,no.57,no.55
    feature_list.extend([angle_based_on_index(shape,49,57,55)])
    
    # feature 4 angle of the nose bridge
    [x1,y1] = shape[27]-shape[30]
    feature_list.extend([cal_angle(x1,y1,1,0)])
    
    # feature 5 angle no.59, no.49, no.51
    feature_list.extend([angle_based_on_index(shape,59,49,51)])
    
    # feature 6 angle no.57, no.55, no.53
    feature_list.extend([angle_based_on_index(shape,57,55,53)])
    
    # feature 7 angle no.49, no.34, no.55
    feature_list.extend([angle_based_on_index(shape,49,34,55)])
    
    #feature 8,9,10 ratio between nose to both cheek
    feature_list.extend([dist.euclidean(shape[0],shape[30])/dist.euclidean(shape[16],shape[30])])
    feature_list.extend([dist.euclidean(shape[3],shape[30])/dist.euclidean(shape[13],shape[30])])
    feature_list.extend([dist.euclidean(shape[5],shape[30])/dist.euclidean(shape[11],shape[30])])
    
    return feature_list
    