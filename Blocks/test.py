import cv2
import numpy as np 
import argparse
import time

model_path = 'D:/WORK/PYTHON/Projects/Smart_Glasses/Smart_Glasses/model_data/yolov3.weights' # model path or trained weights path
anchors_path = 'D:/WORK/PYTHON/Projects/Smart_Glasses/Smart_Glasses/model_data/yolo_anchors.txt'
classes_path = 'D:/WORK/PYTHON/Projects/Smart_Glasses/Smart_Glasses/model_data/coco_classes.txt'
cfg_path =  "D:/WORK/PYTHON/Projects/Smart_Glasses/Smart_Glasses/model_data/yolov3.cfg"

def start_video(video_path):
	model, classes, colors, output_layers = load_yolo()
	cap = cv2.VideoCapture(video_path)
	while True:
		_, frame = cap.read()
		height, width, channels = frame.shape
		blob, outputs = detect_objects(frame, model, output_layers)
		boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
		draw_labels(boxes, confs, colors, class_ids, classes, frame)
		key = cv2.waitKey(1)
		if key == 27:
			break
	cap.release()
   

def load_yolo():
	net = cv2.dnn.readNet('D:/WORK/PYTHON/Projects/Smart_Glasses/Smart_Glasses/model_data/yolov3.weights', "D:/WORK/PYTHON/Projects/Smart_Glasses/Smart_Glasses/model_data/yolov3.cfg")
	classes = []
	with open('D:/WORK/PYTHON/Projects/Smart_Glasses/Smart_Glasses/model_data/coco_classes.txt', "r") as f:
		classes = [line.strip() for line in f.readlines()]
	output_layers = [layer_name for layer_name in net.getUnconnectedOutLayersNames()]
	colors = np.random.uniform(0, 255, size=(len(classes), 3))
	return net, classes, colors, output_layers



def load_image(img_path):
	# image loading
	img = cv2.imread(img_path)
	img = cv2.resize(img, None, fx=0.4, fy=0.4)
	height, width, channels = img.shape
	return img, height, width, channels

def detect_objects(img, net, outputLayers):			
	blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
	net.setInput(blob)
	outputs = net.forward(outputLayers)
	return blob, outputs

def draw_labels(boxes, confs, colors, class_ids, classes, img): 
	indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
	font = cv2.FONT_HERSHEY_PLAIN
	for i in range(len(boxes)):
		if i in indexes:
			x, y, w, h = boxes[i]
			label = str(classes[class_ids[i]])
			color = colors[i]
			cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
			cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
	cv2.imshow("Image", img)


def get_box_dimensions(outputs, height, width):
	boxes = []
	confs = []
	class_ids = []
	for output in outputs:
		for detect in output:
			scores = detect[5:]
			#print(scores)
			class_id = np.argmax(scores)
			conf = scores[class_id]
			if conf > 0.3:
				center_x = int(detect[0] * width)
				center_y = int(detect[1] * height)
				w = int(detect[2] * width)
				h = int(detect[3] * height)
				x = int(center_x - w/2)
				y = int(center_y - h / 2)
				boxes.append([x, y, w, h])
				confs.append(float(conf))
				class_ids.append(class_id)
	return boxes, confs, class_ids




start_video("2.MOV")
cv2.destroyAllWindows()
