#!/usr/bin/python  
# -*- coding: utf-8 -*-  
import time
import dlib
import cv2
from imutils import face_utils
from scipy.spatial import distance as dist
 

def cal_raw_closure(frame, detector, predictor,lStart, lEnd, rStart, rEnd):
    avr_closure = -1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale frame
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
        
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        
        # calculate closure
        leftClosure = eye_closure_ratio(leftEye)
        rightClosure = eye_closure_ratio(rightEye) 
        avr_closure = (leftClosure+rightClosure)/2.0
        
        # ----------------------------------------------------------
        # draw contours of the eyes on the screen
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (238,238,0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (238,238,0), 1)
        
    return avr_closure
    
def eye_closure_ratio(eye):
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye closure ratio
    closure = (A + B) / (2.0 * C)

    return closure
        
def update_initial_ratio(temp_mean, eye_ratio):
    THRESHOLD = 0.0005
    flag = False
    if temp_mean < eye_ratio:
        if abs(eye_ratio - temp_mean) < THRESHOLD:
            flag = True
        return flag,temp_mean
    return flag,eye_ratio
        
        
        
        
        
        
        
        
        
        
        
        
        
        