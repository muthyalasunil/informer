import sys

# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import datetime
 
def detectface(image):

	# Get user supplied values
	cascPath = "/home/pi/pycode/haarcascade_frontalface_default.xml"

	# Create the haar cascade
	faceCascade = cv2.CascadeClassifier(cascPath)

	# Read the image
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Detect faces in the image
	faces = faceCascade.detectMultiScale(gray, 1.3, 5)


	if ( len(faces) == 0 ):
		return None

	# Draw a rectangle around the faces
	for (x, y, w, h) in faces:
    		cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

	print("Found {0} faces!".format(len(faces)))
	
	today = datetime.datetime.today()
	timestr = today.strftime("%Y%m%d%H%M")
	cv2.imwrite('data/fimage'+timestr+'.png',image)

