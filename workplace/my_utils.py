#!/usr/bin/python  
# -*- coding: utf-8 -*-  
import dlib
import cv2
from imutils import face_utils
import numpy as np
from scipy.spatial import distance as dist
from my_extractor import extract_features

def draw_landmarks(frame, detector, predictor, clf):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale frame
    label = ['Others','Happy']
    rects = detector(gray, 0)
    if len(rects)>0: 
        # find the largest face
        max_index = 0
        temp_area = 0
        for i in range(len(rects)):
            if rects[i].area()>temp_area:
                max_index = i
                temp_area = rects[i].area()
        rect = rects[max_index]
        
        # find facial landmarks
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        features = extract_features(shape)
        a = clf.predict(np.asarray([features]))
        pro = clf.predict_proba(np.asarray([features]))
        #print(pro)
        
        # draw the face
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
         # draw the sign
        ty = y - 15 if y - 15 > 15 else y + 15
        cv2.putText(frame, label[a[0]] + '(h:%s, o:%s)'%(str(pro[0][1]),str(pro[0][0])), 
                        (x, ty), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255,255,255), 2)
        # draw the landmarks
        for (x,y) in shape:
            cv2.circle(frame, (x,y), 1, (238,238,0), -1)
            
       
            
    return frame
 
def App_detect_happy(frame, detector, predictor, clf):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale frame
    label = ['Others','Happy']
    rects = detector(gray, 0)
    if len(rects)>0: 
        # find the largest face
        max_index = 0
        temp_area = 0
        for i in range(len(rects)):
            if rects[i].area()>temp_area:
                max_index = i
                temp_area = rects[i].area()
        rect = rects[max_index]
        
        # find facial landmarks
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        features = extract_features(shape)
        a = clf.predict(np.asarray([features]))
            
    return a[0]    

    
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