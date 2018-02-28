#!/usr/bin/python  
# -*- coding: utf-8 -*-  

# My functions
import dlib
import cv2
import time
import heapq
import numpy as np
from imutils.video import VideoStream
from imutils import face_utils
import model_check
import my_utils


# --------------------- Main func ------------------------------------------
def main():
    # check existence of 68 face landmarks model
    file_name = "shape_predictor_68_face_landmarks.dat"
    model_check.model_check(file_name)
    
    # Loading model
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
    # Start video thread
    print("[INFO] starting video stream thread...")
    webcam = 0
    video = VideoStream(webcam).start()
    time.sleep(1.0)

    # facial landmarks for the left and right eye, respectively
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    
    # Mode switch: 
    # ---mode 1: initial parameters
    # ---mode 2: detection and alarm
    mode = 1
    # INITIALIZATION: open and close eye ratio
    open_eye_ratio = -1
    close_eye_ratio = 10
    open_eye_ratio_DONE = False
    close_eye_ratio_DONE = False
    # INITIALIZATION: min_heap, max_heap
    min_heap = []
    max_heap = []
    
    frame_counter = 0
    
    # process each frame
    while True:
        frame = video.read()
        raw_closure = my_utils.cal_raw_closure(frame, detector, predictor,
                                lStart, lEnd, rStart, rEnd)
        
        if raw_closure!=-1:
            # [TEST]: output raw closure
            #print(raw_closure)
            
            if mode == 1:
                # adjusting parameters of closure ratio
                title = "Please blink your eyes in front of the webcam"
                cv2.putText(frame, title, (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                              
                heapq.heappush(min_heap, -raw_closure) # close eye
                heapq.heappush(max_heap, raw_closure)  # open eye
                if len(min_heap)>10:
                    heapq.heappop(min_heap)
                if len(max_heap)>10:
                    heapq.heappop(max_heap)
                
                # first 100 frames are used to initial the two heap
                if frame_counter < 100:
                    frame_counter += 1
                # find if the maximum and minimum data become stable
                else:
                    if open_eye_ratio_DONE == False:
                        open_eye_ratio_DONE, open_eye_ratio = \
                              my_utils.update_initial_ratio(-np.mean(max_heap),-open_eye_ratio)
                        open_eye_ratio = -open_eye_ratio
                        cv2.putText(frame, "open_eye_ratio: Pending", (400, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    else:
                        cv2.putText(frame, "open_eye_ratio: Done", (400, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                              
                    if close_eye_ratio_DONE == False:
                        close_eye_ratio_DONE, close_eye_ratio = \
                              my_utils.update_initial_ratio(-np.mean(min_heap),close_eye_ratio)
                        cv2.putText(frame, "clos_eye_ratio: Pending", (400, 90),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    else:
                        cv2.putText(frame, "clos_eye_ratio: Done", (400, 90),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                   
                    if (close_eye_ratio_DONE==True) and (open_eye_ratio_DONE==True):
                        mode = 2

            elif mode == 2:
                #
                cv2.putText(frame, "clos_eye_ratio: Done", (400, 90),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, "open_eye_ratio: Done", (400, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
        # show the frame
        cv2.imshow("Frame", frame)
        
        # if the `q` key was pressed, break from the loop
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        
    # clean up the windows and close video
    cv2.destroyAllWindows()
    video.stop()
    print("[INFO] App exited")
    
    


# --------------------- Start point ------------------------------------------
if __name__ == "__main__":
    main()



