import cv2
import numpy as np 
import argparse
import time
from PIL import Image
import os


class YOLO(object):
	def __init__(self):
		self.model_path = 'D:/WORK/PYTHON/Projects/TECHNICAL-SOLUTIONS-FOR-VISUALLY-IMPAIRED-master/TECHNICAL-SOLUTIONS-FOR-VISUALLY-IMPAIRED-master/Blocks/model_data/yolov3.weights' # model path or trained weights path
		self.anchors_path = 'D:/WORK/PYTHON/Projects/TECHNICAL-SOLUTIONS-FOR-VISUALLY-IMPAIRED-master/TECHNICAL-SOLUTIONS-FOR-VISUALLY-IMPAIRED-master/Blocks/model_data/yolo_anchors.txt'
		self.classes_path = 'D:/WORK/PYTHON/Projects/TECHNICAL-SOLUTIONS-FOR-VISUALLY-IMPAIRED-master/TECHNICAL-SOLUTIONS-FOR-VISUALLY-IMPAIRED-master/Blocks/model_data/coco_classes.txt'
		self.cfg_path =  "D:/WORK/PYTHON/Projects/TECHNICAL-SOLUTIONS-FOR-VISUALLY-IMPAIRED-master/TECHNICAL-SOLUTIONS-FOR-VISUALLY-IMPAIRED-master/Blocks/model_data/yolov3.cfg"
		self.scalefactor=0.00392
		self.faceDetector=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')  
		self.cam=cv2.VideoCapture(0,cv2.CAP_DSHOW)
		self.recognizer= cv2.face.LBPHFaceRecognizer_create()
		self.path='dataset'\
		#self.score = 0.3
		#self.iou = 0.45
		#self.class_names = self._get_class()
		#self.anchors = self._get_anchors()
		#self.sess = K.get_session()
		#self.model_image_size = (416, 416) # fixed size or (None, None), hw
		#self.boxes, self.scores, self.classes = self.generate()
	
	
	def Test_Camera(self):
		while True:
			ret,img=self.cam.read()
			face,Index=self._get_face_image_cv(img,self.faceDetector)
			if face is not None:
				(x,y,w,h)=Index
				cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,0),2)
			cv2.imshow("Face",img)
			if(cv2.waitKey(1)==ord('q')):
				break
		cv2.destroyAllWindows() 

	def register(self,id):
		if not os.path.exists("dataset"):
			os.makedirs("dataset")
		sampleNum=0
		while True:
			ret,img=self.cam.read()
			face,Coords=self._get_face_image_cv(img,self.faceDetector)
			if face is not None:
				(x,y,w,h)=Coords
				sampleNum=sampleNum+1
				writepath="dataset/user."+str(id)+"."+str(sampleNum)+".jpg"
				im_pil = Image.fromarray(face)
				im_pil.save(writepath)
                #cv2.imwrite("dataset/user."+str(id)+"."+str(sampleNum)+".jpg",face)
				cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,0),2)
				cv2.waitKey(100)
			cv2.imshow("Face",img)
			cv2.waitKey(1)
			if(sampleNum>20):
				break
		cv2.destroyAllWindows()

	def training(self):
		Ids, faces=self._getImageWithID(self.path)
		self.recognizer.train(faces,Ids)
		self.recognizer.save('recognizer/trainingData.yml')

	def detect(self):
		self.recognizer.read("recognizer/trainingData.yml")
		id=0
		ids=[]
		font=cv2.FONT_HERSHEY_COMPLEX
		sampleNum=0

		while True:
			ret,img=self.cam.read()
			face,Coords=self._get_face_image_cv(img,self.faceDetector)
			if face is not None:
				sampleNum=sampleNum+1
				(x,y,w,h)=Coords
				cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,0),2)
				id,conf=self.recognizer.predict(face)
				ids.append(id)
				cv2.putText(img,str(id),(x,y+h),font,1,(255,0,0),2)
				cv2.imshow("Face",img)
			cv2.waitKey(1)
			if(sampleNum>20):
				break

		cv2.destroyAllWindows()
		return max(set(ids), key = ids.count)


	def webcam_detect(self):
		model, classes, colors, output_layers = self.load_yolo()
		while (self.cam.isOpened()):
			grabbed, frame = self.cap.read()
			if grabbed:
				height, width, channels = frame.shape
				blob, outputs = self.detect_objects(frame, model, output_layers)
				boxes, confs, class_ids = self.get_box_dimensions(outputs, height, width)
				self.draw_labels(boxes, confs, colors, class_ids, classes, frame)
				key = cv2.waitKey(1)
				if key == 27:
					break
			else:
				break
		cv2.destroyAllWindows()




	#Load yolo
	def load_yolo(self):
		net = cv2.dnn.readNet(self.model_path,self.cfg_path)
		classes = []
		with open(self.classes_path, "r") as f:
			classes = [line.strip() for line in f.readlines()] 
		
		output_layers = [layer_name for layer_name in net.getUnconnectedOutLayersNames()]
		colors = np.random.uniform(0, 255, size=(len(classes), 3))
		return net, classes, colors, output_layers

	


	

	@staticmethod
	def display_blob(blob):
		'''
			Three images each for RED, GREEN, BLUE channel
		'''
		for b in blob:
			for n, imgb in enumerate(b):
				cv2.imshow(str(n), imgb)

	def detect_objects(self,img, net, outputLayers):			
		blob = cv2.dnn.blobFromImage(img, scalefactor=self.scalefactor, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
		net.setInput(blob)
		outputs = net.forward(outputLayers)
		return blob, outputs

	@staticmethod
	def get_box_dimensions(outputs, height, width):
		boxes = []
		confs = []
		class_ids = []
		for output in outputs:
			for detect in output:
				scores = detect[5:]
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


	@staticmethod			
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



	@staticmethod
	def _getImageWithID(path):
		imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
		faces=[]
		IDs=[]
		for imagePath in imagePaths:
			faceImg=Image.open(imagePath).convert('L')
			faceNp=np.array(faceImg)
			ID=int(os.path.split(imagePath)[-1].split('.')[1])
			faces.append(faceNp)
			IDs.append(ID)
            #cv2.imshow("training",faceNp)
            #cv2.waitKey(10)
		return np.array(IDs), faces

	
	@staticmethod
	def _get_face_image_cv(camera_image,face_detector):
		img_gray = cv2.cvtColor(camera_image, cv2.COLOR_BGR2GRAY, 1)
		faces = face_detector.detectMultiScale(img_gray,1.3,5)

        # Crop the first face found
		if len(faces):
			x, y, w, h = faces[0].tolist()
			face_image = img_gray[y:y + h, x:x + w]
			return (face_image, (x, y, w, h))

		return (None, None)


	def __del__(self):
    	#self.vreader.close()
		self.cam.release()

	"""def load_image(img_path):
			# image loading
			img = cv2.imread(img_path)
			img = cv2.resize(img, None, fx=0.4, fy=0.4)
			height, width, channels = img.shape
			return img, height, width, channels"""


	"""def image_detect(img_path): 
		model, classes, colors, output_layers = self.load_yolo()
		image, height, width, channels = self.load_image(img_path)
		blob, outputs = self.detect_objects(image, model, output_layers)
		boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
		draw_labels(boxes, confs, colors, class_ids, classes, image)
		while True:
			key = cv2.waitKey(1)
			if key == 27:
				break"""

	


	"""def start_video(video_path):
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
		cap.release()"""



if __name__ == '__main__':
	#webcam = args.webcam
	webcam = 1
	Y=YOLO()
	#video_play = args.play_video
	#image = args.image
	if webcam:
		#if args.verbose:
		#	print('---- Starting Web Cam object detection ----')
		Y.webcam_detect()
	"""if video_play:
		video_path = args.video_path
		if args.verbose:
			print('Opening '+video_path+" .... ")
		start_video(video_path)
	if image:
		image_path = args.image_path
		if args.verbose:
			print("Opening "+image_path+" .... ")
		image_detect(image_path)"""
	

	cv2.destroyAllWindows()