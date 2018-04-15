#!/usr/bin/python  
# -*- coding: utf-8 -*-  
import dlib
import cv2
import numpy as np
import time
from imutils import face_utils
import my_utils
from sklearn.externals import joblib
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--mode",type=int, required=True,
        help="mode 1 for webcam, mode 2 for image")
    ap.add_argument("-w", "--webcam", type=int,default=0,
        help="webcam name")
    ap.add_argument("-p", "--path", type=str, default='.\\testImage\\3.jpg',
        help="choose the name of the image")
    args = vars(ap.parse_args())

    file_name = "..\\model\\shape_predictor_68_face_landmarks.dat"
    # Loading model
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(file_name)

    
    if (args['mode'] == 1):
        srcImg = cv2.imread(args['path'])
        srcImg = my_utils.draw_landmarks(srcImg, detector, predictor)
        cv2.imshow("image", srcImg)
        cv2.waitKey(0)
        
    elif (args['mode'] == 2):
        # Start video thread
        print("[INFO] starting video stream thread...")
        webcam = args['webcam']
        video = cv2.VideoCapture(webcam)
        time.sleep(1.0)
        
        # ------------------ time ---------------------
        start = time.clock()
        end = time.clock()
        framecounter = 0
        # ----------------------------------------------
        
        record1 = []
        record1.extend([[0,n] for n in range(36,48)])
        record1.extend([[0,n] for n in range(31,36)])
        record1.extend([[0,n] for n in range(48,60)])
       
        record2 = []
        record2.extend([[0,n] for n in range(36,48)])
        record2.extend([[0,n] for n in range(31,36)])
        record2.extend([[0,n] for n in range(48,60)])
        
        ranking1 = np.array([[0.0,n] for n in range(68)])
        ranking2 = np.array([[0.0,n] for n in range(68)])
        # process each frame
        while True:
            ret, frame = video.read()
            if ret == False:
                break
            
            
            frame,records1,records2,ranking1,ranking2 = my_utils.draw_landmarks(frame, detector, predictor,record1,record2,ranking1,ranking2)
            
            record1 = records1
            record2 = records2
            
            cv2.imshow("capture", frame)
            
            # ------------------ time ---------------------
            # calculate number of frames per second
            framecounter += 1
            end = time.clock()
            if end-start > 1:
                #print(framecounter)
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