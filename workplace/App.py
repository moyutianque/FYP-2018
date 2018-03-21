#!/usr/bin/python  
# -*- coding: utf-8 -*-  
import dlib
import cv2
import numpy as np
import time
from imutils import face_utils
import my_utils
from sklearn.externals import joblib

def main():
    file_name = "shape_predictor_68_face_landmarks.dat"
    m_path = ".\\training\\model\\learned_model.pkl"
    # Loading model
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    print("[INFO] loading expression model")
    clf = joblib.load(m_path)
     
    # Start video thread
    print("[INFO] starting video stream thread...")
    webcam = 0
    video = cv2.VideoCapture(webcam)
    time.sleep(1.0)
    
    # ------------------ time ---------------------
    start = time.clock()
    end = time.clock()
    framecounter = 0
    # ----------------------------------------------
    
    label = 0
    counter = 0
    # process each frame
    while True:
        ret, frame = video.read()
        if ret == False:
            break
        
        label = my_utils.App_detect_happy(frame, detector, predictor,clf)
        cv2.imshow("capture", frame)
        if label==1:
            cv2.imwrite(".\\output\\%d.bmp"%(counter),frame)
            counter+=1
        
        # ------------------ time ---------------------
        # calculate number of frames per second
        framecounter += 1
        end = time.clock()
        if end-start > 1:
            print(framecounter)
            framecounter = 0
            start = time.clock()
        # ----------------------------------------------
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()