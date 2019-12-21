import numpy as np
import cv2 as cv 
import time
import imutils
import RPi.GPIO as GPIO
 
from imutils import contours
from scipy.spatial imporstance as dist
from collections import OrderedDict
from picamera.array import PiRGBArray
from picamera import PiCamera
from subprocess import call  


#gpio init 
GPIO.setmode(GPIO.BCM)
indice_pins = (5,6,13,19)
for pins in indice_pins:
    GPIO.setup(pins, GPIO.OUT)
    GPIO.output(pins, 0)
GPIO.setup(26, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(20, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)


#class and function definitions 

#this functions is used to control the level of light in the picture 
def adjust_gamma(image,gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i/255.0)** invGamma)*255
                      for i in np.arange(0,256)]).astype("uint8")
    return cv.LUT(image,table)

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv.Canny(image, lower, upper)
	# return the edged image
	return edged
# a class to use multi threading in roder to improve the performance
class PiVideoStream:
	def __init__(self, resolution=(320, 240), framerate=32):
		# initialize the camera and stream
		self.camera = PiCamera()
		self.camera.resolution = resolution
		self.camera.framerate = framerate
		self.rawCapture = PiRGBArray(self.camera, size=resolution)
		self.stream = self.camera.capture_continuous(self.rawCapture,
	 		format="bgr", use_video_port=True)
         
		# initialize the frame and the variable used to indicate
		# if the thread should be stopped
		self.frame = None
		self.stopped = False
	def start(self):
		# start the thread to read frames from the video stream
		Thread(target=self.update, args=()).start()
		return self
	def update(self):
		# keep looping infinitely until the thread is stopped
		for f in self.stream:
			# grab the frame from the stream and clear the stream in
			# preparation for the next frame
			self.frame = f.array
			self.rawCapture.truncate(0)
 
			# if the thread indicator variable is set, stop the thread
			# and resource camera resources
			if self.stopped:
				self.stream.close()
				self.rawCapture.close()
				self.camera.close()
				return
	def read(self):
		# return the frame most recently read
		return self.frame
 
	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True
        
# this class helps to label the colour of a given contour 
class ColorLabeler:
	def __init__(self):
		colors = OrderedDict({
			"yellow": (247,181, 0),
			"green": (97, 153, 59),
			"black": (14, 14, 16),
			"blue": (0, 124, 176),
			"orange": (208, 93, 40),})

		self.lab = np.zeros((len(colors), 1, 3), dtype="uint8")
		self.colorNames = []
		for (i, (name, rgb)) in enumerate(colors.items()):
			self.lab[i] = rgb
			self.colorNames.append(name)
		self.lab = cv.cvtColor(self.lab, cv.COLOR_RGB2LAB)
	def label(self,image,c):
		mask = np.zeros(image.shape[:2], dtype="uint8")
		cv.drawContours(mask, [c], -1, 255, -1)
		kernel = np.ones((2,2),np.uint8)
		mask = cv.erode(mask, kernel, iterations=2)
		mean = cv.mean(image, mask=mask)[:3]
 
		minDist = (np.inf, None)
 
		for (i, row) in enumerate(self.lab):
			d = dist.euclidean(row[0], mean)
 			if d < minDist[0]:
				minDist = (d, i)
		return (self.colorNames[minDist[1]],d)
# this class we can determine whether the countour is a rectangle or not 
class ShapeDetector:
	def __init__(self):
		pass 
	def detect(self,c):
		shape = False
		peri = cv.arcLength(c,True)
		approx  = cv.approxPolyDP(c,0.01 * peri ,True)
		if len(approx) == 4 and len(approx) <=6:
			(x, y, w, h) = cv.boundingRect(approx)
			aspectRatio = w / float(h)

			area = cv.contourArea(c)
			hullArea = cv.contourArea(cv.convexHull(c))
			solidity = area / float(hullArea)

			keepDims = w> 25 and h > 25
			keepSolidity = solidity >0.9
			keepAspectRatio = aspectRatio >= 0.8 and aspectRatio <= 1.2
			if keepDims and keepSolidity and keepAspectRatio :
                                shape = True
		return shape
#class instanciation 
sd = ShapeDetector()
cl = ColorLabeler()
vs = PiVideoStream().start()
time.sleep(1.0)
#variables ,tuples and dictionaries 
data_collected = np.rec.array([(0.,0.,0.,'0'),(0.,0.,0.,'0'),(0.,0.,0.,'0')],dtype=[('cX','f2'),('cY','f2'),('prec','f2'),('color','S7')])
test_data=np.rec.array([(0.,0.,0.,'0'),(0.,0.,0.,'0'),(0.,0.,0.,'0')],dtype=[('cX','f2'),('cY','f2'),('prec','f2'),('color','S7')])

plates = [ ["green","black","orange"],
           ["blue","black","yellow"],
           ["orange","green","blue"],
           ["black","green","yellow"],
           ["orange","yellow","black"],
           ["blue","yellow","green"],
           ["yellow","orange","blue"],
           ["yellow","orange","green"],
           ["green","blue","black"],
           ["yellow","blue","orange"],
           ["orange","black","green"],
           ["yellow","black","blue"],
           ["blue","green","orange"],
           ["yellow","green","black"],
           ["black","yellow","orange"],
           ["green","yellow","blue"],
           ["blue","orange","yellow"],
           ["green","orange","yellow"],
           ["black","blue","green"],
           ["orange","blue","yellow"]
           ]


num = {
      '0':(0,0,0,0)
      '1':(0,0,0,1)
      '2':(0,0,1,0)
      '3':(0,0,1,1)
      '4':(0,1,0,0)
      '5':(0,1,0,1)
      '6':(0,1,1,0)
      '7':(0,1,1,1)
      '8':(1,0,0,0)
      '9':(1,0,0,1)
      '10':(1,0,1,0)
        }
# a counter to check the nimber of correct colours in a percieved plate 
maxCmp = 2
while(1):
    frame =vs.read
	i=0
	indice = 0
	maxCmp= 2
    """ apply the filters and blurring on image in order to impove the results of findCountours function"""
	#gamma = 0.9
    #frame2 = adjust_gamma(frame ,gamma=gamma)
    blurred =cv.GaussianBlur(frame, (7,7), 0)
	lab = cv.cvtColor(blurred, cv.COLOR_BGR2LAB)
	wide = cv.Canny(blurred, 50, 50)
	kernel = np.ones((5,5),np.uint8)
	wide = cv.morphologyEx(wide, cv.MORPH_CLOSE, kernel)
	cnts = cv.findContours(wide.copy(), cv.RETR_TREE,
                cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
	(cnts, _) = contours.sort_contours(cnts)
    """determine id the shape is a rectangle or not and store its color""" 
	for c in cnts:   
		if (sd.detect(c) == True) and  (i <3):
			M = cv.moments(c)
			cX = int((M["m10"] / (M["m00"])) )
			cY = int((M["m01"] / (M["m00"])) )
			color,d = cl.label(lab,c)
			test_data[i].cX = cX
			test_data[i].cY = cY
			test_data[i].prec = d
			test_data[i].color = color
			i=i+1    
   
	if  i == 3  :
    """determine if the data we just read is better than the data we have """
        for j in range(3):
            if (test_data[j].prec < data_collected[j].prec or test_data[j].color != data_collected[j].color):
                data_collected[j].cX = test_data[j].cX
                data_collected[j].cY = test_data[j].cY
                data_collected[j].prec = test_data[j].prec
                data_collected[j].color = test_data[j].color
                flag = True
                obtained_colors = data_collected['color']
     """determine whether the plate we read is a plate we know or not """ 
    for j in range(20):
        cmpt = 0
        if (plates[j][0] == obtained_colors[0]) :
            cmpt =cmpt+1
        if (plates[j][1] == obtained_colors[1]) :
            cmpt =cmpt+1
        if (plates[j][2] == obtained_colors[2]) :
            cmpt =cmpt+1
        if cmpt >= maxCmp:
            maxCmp = cmpt
            indice = j+1
     """considering that we take the plates and the plates turned so we have 20 possibilities and we only need 10 so we just remove 10"""
    if indice>10 and indice != 0 :
        indice = indice - 10
    print(indice)
    """configure the ouputs to print the number as a binary number"""
    str_indice = str(indice)
    for pin in range(0,4)
        GPIO.output(indice_pins[pin],num[str_indice][pin])
        time.sleep(0.001)
        
	cv.imshow("window", frame)
	rawCapture.truncate(0)
	if cv.waitKey(1) & 0xFF == ord('q'):
        	break
 #elif  not (GPIO.input(26)) and  GPIO.input(20):
     #GPIO.cleanup()
     #break

cv.destroyAllWindows()
#call("sudo shutdown -h now", shell=True)
