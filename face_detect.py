import sys

# import the necessary packages
import time
import cv2
import datetime
import os
import glob
 
import sys
sys.path.append('pyimagesearch')
from tempimage import TempImage


def detectface(conf, oframe):

        image = cv2.resize(oframe, (oframe.shape[1] / 2, oframe.shape[0] / 2))

	t = TempImage('face') 
	cv2.imwrite(t.path,image)
	# Read the image
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	cascPath = conf["cascPath"] + '/*face*.xml'
	for cascfile in glob.glob(cascPath):
		faceCascade = cv2.CascadeClassifier(cascfile)
		# Detect faces in the image
		faces = faceCascade.detectMultiScale(gray, 1.3, 5)

		if ( len(faces) == 0 ):
			print('no faces found!')
			continue	

		# Draw a rectangle around the faces
		for (x, y, w, h) in faces:
    			cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

		print("Found {0} faces!".format(len(faces)))
		t = TempImage('face') 
		cv2.imwrite(t.path,image)
		
		return
def run():
	conf={}
	conf['cascPath']='/home/pi/opencv-3.4.0/data/haarcascades'
	for filename in glob.glob('data/*.png'):
		detectface(conf, cv2.imread(filename))

