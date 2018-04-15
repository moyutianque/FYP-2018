#!/usr/bin/python  
# -*- coding: utf-8 -*- 
import numpy as np
from scipy.spatial import distance as dist
import itertools
import cv2
import math

"""
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
 """   

def extract_features(shape):
    feature_list_v = []
    feature_list_h = []
    [x1,y1] = shape[27]-shape[8]
    angle = cal_angle(x1,y1,1,0)
    w_v = np.sin(angle)
    w_h = np.cos(angle)
    pivot = []
    pivot.extend([n for n in range(36,48)])
    pivot.extend([n for n in range(31,36)])
    pivot.extend([n for n in range(48,60)])

    # vertical features
    top = 19
    bottom = 8
    left = 0
    right = 16
    for pt in pivot:
        
        feature = (abs(shape[pt][1]-shape[top][1]) * w_v)/(abs(shape[bottom][1]-shape[top][1]) * w_v)
        
        feature_list_v.append(feature)
        feature2 = (abs(shape[pt][0]-shape[left][0])*w_h)/(abs(shape[right][1]-shape[top][1])*w_h)
        feature_list_h.append(feature2)
    return feature_list_v,feature_list_h
   
def angle_based_on_index(shape,idx1,idx2,idx3):
    [x1,y1] = shape[idx1-1]-shape[idx2-1]
    [x2,y2] = shape[idx3-1]-shape[idx2-1]
    return cal_angle(x1,y1,x2,y2)
    
    
def cal_angle(x1, y1, x2, y2):
    denominator = np.sqrt(np.power(x1,2)+np.power(y1,2))*np.sqrt(np.power(x2,2)+np.power(y2,2))
    cos_angle = (x1*x2+y1*y2)/denominator
    
    if cos_angle>1:
        cos_angle=1
    elif cos_angle<-1:
        cos_angle = -1
        
    angle = np.arccos(cos_angle)
    while (angle<0):
        angle += np.pi
    while (angle>np.pi):
        angle -= np.pi
    #angle = angle*360/2/np.pi
    return angle
    
# -------------------------------------head rotation --------------------------
# 3D model points (arbitrary reference).
model_points = np.array([
                        (0.0, 0.0, 0.0),             # Nose tip
                        (0.0, -330.0, -65.0),        # Chin
                        (-225.0, 170.0, -135.0),     # Left eye left corner
                        (225.0, 170.0, -135.0),      # Right eye right corne
                        (-150.0, -150.0, -125.0),    # Left Mouth corner
                        (150.0, -150.0, -125.0)      # Right mouth corner
                     
                    ])
def head_rotation(shape,rect):
    image_points = np.array([shape[33],shape[8],shape[45],shape[36],shape[54],shape[48]],dtype="double")
    size = [rect.height(),rect.width()]
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )
    
    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, 
                    image_points, camera_matrix, dist_coeffs, 
                    flags=cv2.SOLVEPNP_ITERATIVE)
 
    return rotation_vector