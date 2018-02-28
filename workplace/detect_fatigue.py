#!/usr/bin/python  
# -*- coding: utf-8 -*-  

import dlib
import cv2
import time
from imutils.video import VideoStream
from imutils import face_utils
from scipy.spatial import distance as dist

def eye_closure_ratio(eye):
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye closure ratio
    closure = (A + B) / (2.0 * C)

    return closure

def normalize_closure(closure, maxRatio, minRatio):
    return (closure-minRatio)*1.0/(maxRatio-minRatio)











def start(webcam):
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # facial landmarks for the left and right eye, respectively
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    
    print("[INFO] starting video stream thread...")
    video = VideoStream(webcam).start()
    time.sleep(1.0)
    
    # INITIALIZATION: mode
    # 0 -- represent adjusting parameter phase
    # 1 -- represent formal detection and alarm phase
    mode = 0
    # INITIALIZATION: submode
    # 0 -- represent adjusting open eye parameter
    # 1 -- represent adjusting open close parameter
    # 2 -- represent waiting phase
    submode = 0
    # INITIALIZATION: open and close eye ratio
    openEyeRatio = -1
    closeEyeRatio = 10
    
    # process each frame
    while True:
        frame = video.read()
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
            leftClosure = eye_closure_ratio(leftEye)
            rightClosure = eye_closure_ratio(rightEye)
            
            avr_closure = (leftClosure+rightClosure)/2.0
            
            ##########################################################
            if mode == 0:
                # adjusting parameter phase
                if submode == 0:
                    # open eye
                    
                elif submode == 2:
                    time.sleep(5.0)
                    submode = 1               
                elif submode == 1:
                    # close eye
                    

            elif mode == 1:
                # formal detection and alarm phase
            
            
            # ----------------------------------------------------------
            # draw contours of the eyes on the screen
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (238,238,0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (238,238,0), 1)
            
             
        # show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
 
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        
    # clean up the windows and close video
    cv2.destroyAllWindows()
    video.stop()
        
        
        
        