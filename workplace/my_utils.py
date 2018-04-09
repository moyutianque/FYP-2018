#!/usr/bin/python  
# -*- coding: utf-8 -*-  
import dlib
import cv2
from imutils import face_utils
import numpy as np
from scipy.spatial import distance as dist
from my_extractor import extract_features
from headrotation import head_rotation
import math

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
        
        """
        features = extract_features(shape)
        a = clf.predict(np.asarray([features]))
        pro = clf.predict_proba(np.asarray([features]))
        """
        
        # draw the face
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
         # draw the sign
        ty = y - 15 if y - 15 > 15 else y + 15
        """
        cv2.putText(frame, label[a[0]] + '(h:%0.2f, o:%0.2f)'%(pro[0][1],pro[0][0]), 
                        (x, ty), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255,255,255), 2)
        """
        [rotation,p1,p2] = head_rotation(shape, frame)
        rotation = [math.degrees(rotation[0,0]),math.degrees(rotation[1,0]),math.degrees(rotation[2,0])]
        cv2.putText(frame,'(yaw:%0.2f,pitch:%0.2f,roll:%0.2f)'%(rotation[1],rotation[0],rotation[2]), 
                        (x, ty), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255,255,255), 2)
        #cv2.line(frame, p1, p2, (255,0,0), 2)
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
