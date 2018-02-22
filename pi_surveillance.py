# USAGE
# python deep_learning_object_detection.py --image images/example_01.jpg \
#	--prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodelc

# import the necessary packages
import sys
sys.path.append('pyimagesearch')

from imutils.video import VideoStream
import cvcamera

from tempimage import TempImage

import argparse
import warnings
import datetime
import imutils
import json
import time
import cv2
import numpy as np

from threading import Thread 

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True,
	help="path to the JSON configuration file")
args = vars(ap.parse_args())

# filter warnings, load the configuration and initialize the Dropbox
# client
warnings.filterwarnings("ignore")
conf = json.load(open(args["conf"]))

cascPath = conf["cascPath"]
print(cascPath)
# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# initialize the camera and grab a reference to the raw camera capture
vs = VideoStream(usePiCamera=True).start()

# allow the camera to warmup, then initialize the average frame, last
# uploaded timestamp, and frame motion counter
print("[INFO] warming up...")
time.sleep(conf["camera_warmup_time"])

def detectface(image):

	print("[INFO] detecting faces...")

	# Read the image
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Detect faces in the image
	faces = faceCascade.detectMultiScale(gray, 1.3, 5)

	print("Found {0} faces!".format(len(faces)))
	if ( len(faces) == 0 ):
		return None

	# Draw a rectangle around the faces
	for (x, y, w, h) in faces:
    		cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

	t = TempImage()
	cv2.imwrite(t.path+'.png',image)



avg = None
lastUploaded = datetime.datetime.now()
motionCounter = 0


while True:
	
	# the timestamp and occupied/unoccupied text
	frame1 = vs.read()
	oframe = vs.read()
	frame2 = vs.read()

	timestamp = datetime.datetime.now()
	text = "Unoccupied"

	# resize the frame, convert it to grayscale, and blur it
	frame = imutils.resize(oframe, width=500)
	frame1 = imutils.resize(frame1, width=500)
	frame2 = imutils.resize(frame2, width=500)

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)

	# if the average frame is None, initialize it
	if avg is None:
		print("[INFO] starting background model...")
		avg = gray.copy().astype("float")
		continue

	# accumulate the weighted average between the current frame and
	# previous frames, then compute the difference between the current
	# frame and running average
	cv2.accumulateWeighted(gray, avg, 0.5)
	frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

	# threshold the delta image, dilate the thresholded image to fill
	# in holes, then find contours on thresholded image
	thresh = cv2.threshold(frameDelta, conf["delta_thresh"], 255,
		cv2.THRESH_BINARY)[1]
	thresh = cv2.dilate(thresh, None, iterations=2)
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]

	# loop over the contours
	for c in cnts:
		# if the contour is too small, ignore it
		if cv2.contourArea(c) < conf["min_area"]:
			continue

		# compute the bounding box for the contour, draw it on the frame,
		# and update the text
		(x, y, w, h) = cv2.boundingRect(c)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		text = "Occupied"

	# draw the text and timestamp on the frame
	ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
	cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
		0.35, (0, 0, 255), 1)

	# check to see if the room is occupied
	if text == "Occupied":
		# check to see if enough time has passed between uploads
		if (timestamp - lastUploaded).seconds >= conf["min_upload_seconds"]:
			# increment the motion counter
			motionCounter += 1

			# check to see if the number of frames with consistent motion is
			# high enough
			if motionCounter >= conf["min_motion_frames"]:
				
				print("[INFO] detected motion...")
                		Thread(detectface(frame)).start()
                		Thread(detectface(frame1)).start()
                		Thread(detectface(frame2)).start()

				lastUploaded = timestamp
				print("[INFO] done capturing frame...")
				
				# counter
				motionCounter = 0

	# otherwise, the room is not occupied
	else:
		motionCounter = 0

	# check to see if the frames should be displayed to screen
	if conf["show_video"]:
		# display the security feed
		cv2.imshow("Security Feed", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key is pressed, break from the lop
		if key == ord("q"):
			break
	
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

