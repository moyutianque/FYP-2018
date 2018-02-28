#!/usr/bin/python  
# -*- coding: utf-8 -*-  

# My functions
import model_check
import detect_fatigue

# --------------------- Main func ------------------------------------------
def main():
    # check existence of 68 face landmarks model
    file_name = "shape_predictor_68_face_landmarks.dat"
    model_check.model_check(file_name)
    
    # main process
    detect_fatigue.start(0)
    
    


# --------------------- Start point ------------------------------------------
if __name__ == "__main__":
    main()



