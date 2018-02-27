#!/usr/bin/python  
# -*- coding: utf-8 -*-  
import os
import sys
import bz2
import urllib.request

# -------------------------- 1. Download 68 face landmarks model ---------------------------
file_name = "shape_predictor_68_face_landmarks.dat"
saveFile = "shape_predictor_68_face_landmarks.dat.bz2"

def report(count, blockSize, totalSize):
	percent = int(count*blockSize*100/totalSize)
	sys.stdout.write(" \r%d%%" % percent + ' Complete')
	sys.stdout.flush()
	
if not os.path.exists(file_name):
	# download
	url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
	sys.stdout.write('\rFetching ' + file_name + '...\n')
	urllib.request.urlretrieve(url, saveFile, reporthook=report)
	sys.stdout.write("\rDownload complete, saved as %s" % (saveFile) + '\n\n')
	sys.stdout.flush()
	
	# decompress
	sys.stdout.write("Decompressing File %s" %(saveFile) + '\n')
	with open(file_name, 'wb') as new_file, bz2.BZ2File(saveFile, 'rb') as file:
		for data in iter(lambda : file.read(100 * 1024), b''):
			new_file.write(data)
else:
	print("FILE: shape_predictor_68_face_landmarks.dat exists")
	
# -------------------------------------------------------------------------------------------





