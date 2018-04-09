#!/usr/bin/python  
# -*- coding: utf-8 -*- 
import os
import dlib
import cv2
import numpy as np
from imutils import face_utils
import csv
from my_extractor import extract_features

file_name = "..\\shape_predictor_68_face_landmarks.dat"
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(file_name)

dir = '.\\training_dataset\\nlaugh\\'
#dir = '.\\hello\\'
output_csv = ".\\train_neg.csv"
with open(output_csv ,"w",newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["ft"]*399)  ## [INFO] revise attention 
    for root,dirs,files in os.walk(dir):
        for file in files:
            print(file)
            srcImg = cv2.imread(str(dir)+file)
            rects = detector(srcImg, 0)
            for rect in rects:
                shape = predictor(srcImg, rect)
                shape = face_utils.shape_to_np(shape)
                # draw the face
                (x, y, w, h) = face_utils.rect_to_bb(rect)
                #cv2.rectangle(srcImg, (x, y), (x + w, y + h), (255, 255, 255), 2)
                # draw the landmarks
                for (x,y) in shape:
                    cv2.circle(srcImg, (x,y), 1, (255,144,30), -1)
            
                writer.writerow(extract_features(shape,rect))
                
            
            # [INFO] comment it down after testing
            #cv2.namedWindow("%s" % {str(file)},cv2.WINDOW_NORMAL)
            #cv2.resizeWindow("%s" % {str(file)}, 640, 480)
            #cv2.imshow("%s" % {str(file)}, srcImg)      
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


    