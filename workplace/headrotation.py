import dlib
import numpy as np
import os
import cv2
import operator
import math
from scipy.spatial import distance as dist
import random
# from pdb import set_trace as bp

# 3D model points (arbitrary reference).
model_points = np.array([
                        (0.0, 0.0, 0.0),             # Nose tip
                        (0.0, -330.0, -65.0),        # Chin
                        (-225.0, 170.0, -135.0),     # Left eye left corner
                        (225.0, 170.0, -135.0),      # Right eye right corne
                        (-150.0, -150.0, -125.0),    # Left Mouth corner
                        (150.0, -150.0, -125.0)      # Right mouth corner
                     
                    ])

def head_rotation(shape,frame):
    image_points = np.array([shape[33],shape[8],shape[45],shape[36],shape[54],shape[48]],dtype="double")
    size = frame.shape
    focal_length = 778#size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )
    
    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, 
                    image_points, camera_matrix, dist_coeffs, 
                    flags=cv2.SOLVEPNP_ITERATIVE)
                        
    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), 
                        rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    p1 = ( int(image_points[0][0]), int(image_points[0][1]))
    p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
 
    #cv2.line(frame, p1, p2, (255,0,0), 2)  
    #print (rotation_vector)
    return [rotation_vector,p1,p2]
    

# Calculates rotation matrix to euler angles
# http://planning.cs.uiuc.edu/node103.html
def rotationMatrixToEulerAngles(R) :
 
    #print (R)
    sy = math.sqrt(R[2,1] * R[2,1] +  R[2,2] * R[2,2])

    yaw = math.atan2(R[1,0] / R[0,0])
    pitch = math.atan2(-R[2,0] / sy)
    row = math.atan2(R[2,1] / R[2,2])

    return np.array([yaw, pitch, row])    
    
    
    
    
    
    
    
    
def head_rotation2(shape, x, y, w, h):
    # focal distance mm->pixel
    # focal = 3.6 mm, 480p (720*480), 1/4 inch
    focal = 778 # approximate
    
    u2 = shape[36]  # leftEyeLeftCorner
    u1 = shape[39]  # leftEyeRightCorner
    v1 = shape[42]  # rightEyeLeftCorner
    v2 = shape[45]  # rightEyeRightCorner
    
    delta_u = dist.euclidean(u1,u2)
    delta_v = dist.euclidean(v1,v2)
    dist_u1v1 = dist.euclidean(u1,v1)
    dist_u2v2 = dist.euclidean(u2,v2)
    dist_u1v2 = dist.euclidean(u1,v2)
    dist_u2v1 = dist.euclidean(u2,v1)
    
    I1 = (delta_u * delta_v)/(dist_u1v2 * dist_u2v1)
    Q = 1/np.sqrt(I1) - 1
    
    A = delta_u/delta_v + 1
    B = ((2/Q)+2)*(delta_u/delta_v-1)
    C = ((2/Q)+1)*(delta_u/delta_v+1)
    print("%f %f %f %f %f %f" %(Q,A,B,C,delta_u,delta_v))
    
    S_pos = (-B + np.sqrt(B**2-4*A*C))/(2*A)
    S_neg = (-B - np.sqrt(B**2-4*A*C))/(2*A)
    
    M = -1/(2+Q)
    u = (delta_v*delta_u*M*dist_u1v1 - M**2*dist_u2v2*dist_u1v1**2)/(delta_v*(M*dist_u1v1-delta_u))
    
    yaw = np.arctan(focal/((S_pos-1)*u))
    yaw2 = np.arctan(focal/((S_neg-1)*u))
    
    return [math.degrees(yaw),math.degrees(yaw2)]
    
    
    
def head_rotation1(faceLandmarks,x,y,w,h):

        #faceLandmarks = np.array([[p.x, p.y] for p in faceShape.parts()])

        faceLandmarks_regulated = np.array(
            [[p[0] - x, p[1] - y] for p in faceLandmarks])
        faceLandmarks_downscaled = np.array(
            [[int((p[0] - x) / 2), int((p[1] - y) / 2)] for p in faceLandmarks])
# ===========================================================================================
# http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4497208
        f = 778   # focal distance
        # leftMouthCorner = faceLandmarks[48]
        # rightMouthCorner = faceLandmarks[54]
        u2 = faceLandmarks[36]  # leftEyeLeftCorner
        u1 = faceLandmarks[39]  # leftEyeRightCorner
        v1 = faceLandmarks[42]  # rightEyeLeftCorner
        v2 = faceLandmarks[45]  # rightEyeRightCorner

# ===========================================yaw================================================
# http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=557271
# https://www.wolframalpha.com/input/?i=(u1-u2)%2F(v1-v2)%3D-((S-1)*(S-(1%2B2%2FQ))%2F((S%2B1)*(S%2B(1%2B2%2FQ))))+solve+S
# https://link.springer.com/content/pdf/10.1007%2F11540007_35.pdf
        f1 = 778

        I1 = (abs(u2[0] - u1[0]) * abs(v1[0] - v2[0])) / \
            (abs(u2[0] - v1[0]) * abs(u1[0] - v2[0]))
        # (((abs(u2[0] - u1[0]) + abs(v1[0] - v2[0])))**2)

        Q = (1 / math.sqrt(I1)) - 1

        # A = abs(u1[0] - u2[0]) / abs(v1[0] - v2[0]) + 1
        # B = ((2 / Q) + 2) * ((abs(u1[0] - u2[0]) / abs(v1[0] - v2[0])) - 1)
        # C = ((2 / Q) + 1) * ((abs(u1[0] - u2[0]) / abs(v1[0] - v2[0])) + 1)
        if (u1[0] != u2[0] - v1[0] + v2[0]) and (Q + 1 != 0) and (Q * v1[0] - Q * v2[0] + 2 * v1[0] - 2 * v2[0] != 0):
            S = (-math.sqrt(-4 * (Q ** 2) * u1[0] * v1[0] + 4 * (Q ** 2) * u1[0] * v2[0] + 4 * (Q ** 2) * u2[0] * v1[0] - 4 * (Q ** 2) * u2[0] * v2[0] - 8 * Q * u1[0] * v1[0] + 8 * Q * u1[0] * v2[0] + 8 * Q * u2[0] * v1[0] - 8 * Q * u2[0] * v2[0] + u1[0] ** 2 -
                            2 * u1[0] * u2[0] - 2 * u1[0] * v1[0] + 2 * u1[0] * v2[0] + u2[0] ** 2 + 2 * u2[0] * v1[0] - 2 * u2[0] * v2[0] + v1[0] ** 2 - 2 * v1[0] * v2[0] + v2[0] ** 2) - Q * u1[0] + Q * u2[0] + Q * v1[0] - Q * v2[0] - u1[0] + u2[0] + v1[0] - v2[0]) / (Q * (u1[0] - u2[0] + v1[0] - v2[0]))
            M = -1 / (2 + Q)

            u = ((abs(v1[0] - v2[0]) * abs(u1[0] - u2[0]) * M * abs(u1[0] - v1[0])) - ((M**2) * abs(u2[0] - v2[0])
                                                                                       * (abs(u1[0] - v1[0])**2))) / (abs(v1[0] - v2[0]) * (M * abs(u1[0] - v1[0]) - abs(u1[0] - u2[0])))
            # deltaE =

            yaw = math.atan(
                f1 / ((S - 1) * u))
        else:
            yaw = 0
        # Sp = (-B + math.sqrt((B ** 2) - 4 * A * C)) / (2 * A)  # S+
        # Ss = (-B - math.sqrt((B ** 2) - 4 * A * C)) / (2 * A)  # S-

        # M = (abs(v1[0] - v2[0]) * u1[0]) / (abs(u1[0] - v1[0]) * v2[0])
# ===========================================pitch===============================================
        p0 = (44.075 * (abs(u1[0] - u2[0]) + abs(v1[0] - v2[0])) / 2) / 30.885
        p1 = faceLandmarks[33][1] - faceLandmarks[27][1]
        if ((p0**2) * (p1**2) - (f**2) * (p1**2) +
                (f**2) * (p0**2)) >= 0:
            E = (p1**2 + math.sqrt((p0**2) * (p1**2) - (f**2) * (p1**2) +
                                   (f**2) * (p0**2))) * (f / (p0 * ((p1**2) + (f**2))))
        else:
            E = 0
        pitch = math.asin(E)
# ===========================================================================================

        #print(math.degrees(roll), math.degrees(yaw), math.degrees(pitch))
        return [math.degrees(yaw),math.degrees(pitch)]
