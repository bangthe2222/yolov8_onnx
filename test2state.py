import numpy as np
import cv2
import time
import os
import imutils
from models.YOLOv8 import YOLOv8

def letterbox( im, new_shape=(320, 320), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
	""""
		Resize and pad image while meeting stride-multiple constraints
		input:
			im-> array[W,H,3]: input image to resize
			new_shape-> (int,int): output size
			color->(114,114,114) : padding color
		ouput:
			im-> array[W,H,3]: ouput image
			r-> float: scale ratio
			dw-> float: width padding
			dh-> float: hight padding
	"""
	shape = im.shape[:2]  # current shape [height, width]
	if isinstance(new_shape, int):
		new_shape = (new_shape, new_shape)

	# Scale ratio (new / old)
	r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
	if not scaleup:  # only scale down, do not scale up (for better val mAP)
		r = min(r, 1.0)

	# Compute padding
	new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
	dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

	if auto:  # minimum rectangle
		dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

	dw /= 2  # divide padding into 2 sides
	dh /= 2

	if shape[::-1] != new_unpad:  # resize
		im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
	top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
	left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
	im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
	return im
class Test2State:
	"""
		This class can create lable from image
	"""
	def __init__(self) -> None:
		prototxtPath =r"./realtime-object-detection-master/MobileNetSSD_deploy.prototxt.txt"
		weightsPath = r"./realtime-object-detection-master/MobileNetSSD_deploy.caffemodel"
		# initialize the list of class labels MobileNet SSD was trained to detect
		# and generate a set of bounding box colors for each class
		self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
		"dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

		self.COLORS = np.random.uniform(0, 255, size = (len(self.CLASSES), 3))
		# load our serialized model from disk
		print("[INFO] loading model...")
		self.net = cv2.dnn.readNet(prototxtPath,weightsPath)
	def detect(self,frame, width = 720):
		"""
			detect bottle funtion and write yolo format label
			Input:	image -> array(None,None,3) : input image
					class_id -> string number : id of classes
					out_path -> string : output path
					file_name -> string : output image file name
					width -> number : ouput image width
			Ouput: None
		"""
		# resize the video stream window at a maximum width of 500 pixels
		# time.sleep(0.4)
		t = time.time()
		frame = imutils.resize(frame, width=width)
		ori_frame = frame.copy()
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
		# pass the blob through the network and get the detections
		self.net.setInput(blob)
		detections = self.net.forward()
		# loop over the detections
		(startX, startY, endX, endY) = (0,0,0,0)
		for i in np.arange(0, detections.shape[2]):
			global label
			# extract the probability of the prediction
			probability = detections[0, 0, i, 2]
			# filter out weak detections by ensuring that probability is
			# greater than the min probability
			if probability > 0.1:
				idx = int(detections[0, 0, i, 1])
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")
				# draw the prediction on the frame
				label = "{}".format(self.CLASSES[idx])
				cv2.rectangle(frame, (startX, startY), (endX, endY), self.COLORS[idx], 2)
				y = startY - 15 if startY - 15 > 15 else startY + 15
				cv2.putText(frame, label+ ": " + str(probability), (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS[idx], 2)
				if label == "bottle":
					startX = startX -20
					startY = startY - 20
					endX = endX + 20
					endY = endY + 20

					if startX < 0: startX = 0
					if startY < 0: startY = 0
					if endX > width: endX = width
					if endY > width: endY = width
		return startX, endX, startY, endY

if __name__ == "__main__":
	model_path = "./EduBinYolov8_3_2_2023.onnx"
	class_names = ["bottle", "milo", "redbull"]
	
	# Initialize YOLOv8 object detector
	yolov8_detector = YOLOv8(model_path,  # path to onnx model 
							class_names= class_names, # class names
							conf_thres=0.1, # confidence threshold
							iou_thres=0.5   # iou threshold 
							)
	make_label = Test2State()
	path = r"C:\Users\PC\Documents\HCMUT K21\EduBin\DEMO_DAY\DATA\Test Data\\"
	for file in os.listdir(path):
		img_path = path + file
		print(img_path)
		image = cv2.imread(img_path)
	# cap = cv2.VideoCapture(0)
	# while True:
	# 	_, image = cap.read()
		image = letterbox(image, (720,720))
		x1, x2, y1, y2 = make_label.detect(image)
		print(x1, x2, y1, y2)
		if (x1 + x2 + y1+ y2) > 0:
			image = image[y1:y2,x1:x2]
			# image = letterbox(image, (720,720))
			yolov8_detector.detect_objects(image)

			# Get Object ID
			id_out = yolov8_detector.getIdObject()
			print(id_out)

			# Draw detections
			combined_img = yolov8_detector.draw_detections(image)

			# Resize image
			out_img = letterbox(combined_img, (720,720))

			# Show iamge
			cv2.imshow("Output", out_img)
		# else:
		# 	yolov8_detector.detect_objects(image)
		# 	# Get Object ID
		# 	id_out = yolov8_detector.getIdObject()
		# 	print(id_out)

		# 	# Draw detections
		# 	combined_img = yolov8_detector.draw_detections(image)

		# 	# Resize image
		# 	out_img = letterbox(combined_img, (720,720))

		# 	# Show iamge
		# 	cv2.imshow("Output", out_img)
		# cv2.rectangle(image, (x1,y1), (x2,y2), (255,0,255), thickness=2, lineType=cv2.LINE_AA)
		else:
			cv2.imshow("Output", image)
		cv2.waitKey(0)
