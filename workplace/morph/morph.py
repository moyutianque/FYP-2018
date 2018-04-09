# -*- coding: utf-8 -*- 
import numpy as np
import cv2
import dlib
from imutils import face_utils

mw = 465
mh = 623 
standard_model = [
    # face
    [35,317],
    [37,367],
    [44,417],
    [57,465],
    [80,510],
    [109,548],
    [148,581],
    [192,606],
    [239,614], # point chin
    [286,606],
    [330,581],
    [369,548],
    [398,510],
    [421,465],
    [434,417],
    [441,367],
    [443,317],
    # left eyebrow
    [48,225],
    [77,212],
    [109,215],
    [137,226],
    [165,239],
    # right eyebrow
    [313,239],
    [341,226],
    [369,215],
    [401,212],
    [430,225],
    # nose bridge
    [239,238],
    [239,320],
    [239,368],
    [239,411],
    # nose
    [190,416],
    [213,427],
    [239,431],
    [265,427],
    [288,416],
    # left eye
    [83,279],
    [116,254],
    [141,254],
    [175,279],
    [141,291],
    [116,291],
    # right eye
    [303,279],
    [337,254],
    [362,254],
    [395,279],
    [362,291],
    [337,291],
    # mouth
    [161,489],
    [191,479],
    [217,470],
    [239,476],
    [261,470],
    [287,479],
    [317,489],
    
    [294,504],
    [271,517],
    [239,520],
    [207,517],
    [184,504],
    
    [175,491],
    [205,494],
    [239,496],
    [273,494],
    [303,491],
    
    [273,494],
    [239,496],
    [205,494]
]

def perfectFace(srcImg, imgMorph, triSrc, triDst):
    rect_triSrc = cv2.boundingRect(np.float32([triSrc]))
    rect_triDst = cv2.boundingRect(np.float32([triDst]))
    size = (rect_triDst[2],rect_triDst[3])
    # triangle points relative distance towards the rectangle
    t_Rsrc = []
    t_Rdst = []
    for i in range(0, 3):
        t_Rsrc.append(((triSrc[i][0] - rect_triSrc[0]),(triSrc[i][1] - rect_triSrc[1])))
        t_Rdst.append(((triDst[i][0] - rect_triDst[0]),(triDst[i][1] - rect_triDst[1])))
    # crop the image according to the position of the rectangle
    srcImg = srcImg[rect_triSrc[1]:rect_triSrc[1] + rect_triSrc[3], rect_triSrc[0]:rect_triSrc[0] + rect_triSrc[2]]
    
    transMatrix = cv2.getAffineTransform(np.float32(t_Rsrc), np.float32(t_Rdst))
    dst_rectImg = cv2.warpAffine( srcImg, transMatrix, (size[0], size[1]), None, 
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )
    
    # Get mask by filling triangle
    mask = np.zeros((rect_triDst[3], rect_triDst[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(t_Rdst), (1.0, 1.0, 1.0), 16, 0);
    dst_rectImg = dst_rectImg * mask
    
    imgMorph[rect_triDst[1]:rect_triDst[1]+rect_triDst[3], rect_triDst[0]:rect_triDst[0]+rect_triDst[2]] \
            = imgMorph[rect_triDst[1]:rect_triDst[1]+rect_triDst[3], rect_triDst[0]:rect_triDst[0]+rect_triDst[2]] \
            * ( (1.0, 1.0, 1.0) - mask )
    imgMorph[rect_triDst[1]:rect_triDst[1]+rect_triDst[3], rect_triDst[0]:rect_triDst[0]+rect_triDst[2]] \
            = imgMorph[rect_triDst[1]:rect_triDst[1]+rect_triDst[3], rect_triDst[0]:rect_triDst[0]+rect_triDst[2]] \
            + dst_rectImg
            
    return imgMorph
    
def morph(shape,srcImg):
    imgMorph = np.zeros([mh,mw,3], dtype=srcImg.dtype)
    with open("triangle.txt") as file :
        for line in file :
            x,y,z = line.split()
            x = int(x)
            y = int(y)
            z = int(z)
            triSrc = [shape[x-1],shape[y-1],shape[z-1]]
            triDst = [standard_model[x-1],standard_model[y-1],standard_model[z-1]]
            
            imgMorph = perfectFace(srcImg, imgMorph, triSrc, triDst)
    cv2.imshow("hh",np.uint8(imgMorph))
    cv2.waitKey(0)

if __name__ == '__main__':
    filename='morph3.jpg'
    srcImg = cv2.imread(filename)
    
    file_name = "..\\shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(file_name)
    print("[INFO] Model load successfully!")
    rects = detector(srcImg, 0)
    
    for rect in rects:
        shape = predictor(srcImg, rect)
        shape = face_utils.shape_to_np(shape)
        morph(shape,np.float32(srcImg))
        
        
