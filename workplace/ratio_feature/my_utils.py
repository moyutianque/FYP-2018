#!/usr/bin/python  
# -*- coding: utf-8 -*-  
import dlib
import cv2
from imutils import face_utils
import numpy as np
from scipy.spatial import distance as dist
from my_extractor import extract_features
import math
from operator import itemgetter

def draw_landmarks(frame, detector, predictor,record1,record2,ranking1,ranking2):
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
          
        features_v,features_h = extract_features(shape)
        
        for i in range(len(features_v)):
            record1[i][0] = abs(features_v[i]-record1[i][0])
            record2[i][0] = abs(features_h[i]-record2[i][0])
            
        records1 = record1
        records2 = record2
        records1.sort(key=itemgetter(0),reverse=True)
        records2.sort(key=itemgetter(0),reverse=True)
        records1 = np.array(records1)
        records2 = np.array(records2)
        records1.astype(int)
        records2.astype(int)
        
        for i in range(29):
            ranking1[records1[i,1],0] += (i*0.1)
            ranking2[records2[i,1],0] += (i*0.1)
        
        m=sorted(ranking1,key=itemgetter(0),reverse=True)
        n=sorted(ranking2,key=itemgetter(0),reverse=True)
        m = np.array(m)
        n = np.array(n)
        cv2.putText(frame, str(m[:,1]), 
                        (10,20), cv2.FONT_HERSHEY_SIMPLEX,0.3, (0,0,0), 1)
        cv2.putText(frame, str(n[:,1]), 
                        (10,40), cv2.FONT_HERSHEY_SIMPLEX,0.3, (0,0,0), 1)
        # draw the face
        """(x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
        """

        #cv2.line(frame, p1, p2, (255,0,0), 2)
        # draw the landmarks
        for (x,y) in shape:
            cv2.circle(frame, (x,y), 1, (238,238,0), -1)
    return frame,record1,record2,ranking1,ranking2
 
  
