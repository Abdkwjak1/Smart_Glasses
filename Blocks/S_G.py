from cProfile import label
from tracemalloc import stop
import cv2
import numpy as np 
import argparse
import time
from PIL import Image
import os 
from playsound import playsound
from ArabicOcr import arabicocr
#from character_segmentation import segment
#from segmentation import extract_words
#from train import prepare_char, featurizer
import multiprocessing as mp
import pickle
from gtts import gTTS

def inner():
    raise Exception("FAIL")

def load_model():
	model_name = 'models/2L_NN.sav'
	location = 'models'
	if os.path.exists(location):
		model = pickle.load(open(model_name, 'rb'))
		return model

class S_G(object):
	def __init__(self):
		self.model_path = 'model_data/yolov3.weights' # model path or trained weights path
		self.anchors_path = 'model_data/yolo_anchors.txt'
		self.classes_path = 'model_data/coco_classes.txt'
		self.cfg_path =  "model_data/yolov3.cfg"
		self.scalefactor=0.00392
		self.faceDetector=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')  
		self.cam=cv2.VideoCapture(0,cv2.CAP_DSHOW)
		self.recognizer= cv2.face.LBPHFaceRecognizer_create()
		self.path='dataset'
		#self.score = 0.3
		#self.iou = 0.45
		#self.class_names = self._get_class()
		#self.anchors = self._get_anchors()
		#self.sess = K.get_session()
		#self.model_image_size = (416, 416) # fixed size or (None, None), hw
		#self.boxes, self.scores, self.classes = self.generate()
	


#Face Recognition Part
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

#Yolo Object Detection Part
	def yolo_detect(self):

		t1 = time.perf_counter()
		model, classes, colors, output_layers = self.load_yolo()
		while (self.cam.isOpened()):
			
			grabbed, frame = self.cam.read()
			b_img = np.zeros((int(self.cam.get(4)),int(self.cam.get(3))),np.uint8)
			if grabbed:
				height, width, channels = frame.shape
				blob, outputs = self.detect_objects(frame, model, output_layers)
				boxes, confs, class_ids = self.get_box_dimensions(outputs, height, width)
				label,pos = self.draw_labels(boxes, confs, colors, class_ids, classes, frame, b_img)
				t2 = time.perf_counter()
				if round(t2-t1,2) > 1000.0:
					t1 = time.perf_counter()
					for i in range(len(label)):
						playsound("sounds/"+label[i]+".mp3")
						playsound("sounds/"+pos[i].split()[-1]+".mp3")

				key = cv2.waitKey(1)
				if key == 27:
					break
			else:
				break
		cv2.destroyAllWindows()

	def load_yolo(self):
		net = cv2.dnn.readNet(self.model_path,self.cfg_path)
		classes = []
		with open(self.classes_path, "r") as f:
			classes = [line.strip() for line in f.readlines()] 
		
		output_layers = [layer_name for layer_name in net.getUnconnectedOutLayersNames()]
		colors = np.random.uniform(0, 255, size=(len(classes), 3))
		return net, classes, colors, output_layers

	

	# Arabic OCR Part 
	def imagetospeech(self):
		i=0
		pre_words=["عيادة","صيدلية","بقالية","مخبر","مستشفى","مشفى","مركز","مكتبة","ماركت"]
		while (self.cam.isOpened()):
			grabbed, frame = self.cam.read()
			if grabbed:
				
				cv2.imshow("test", frame)
				key = cv2.waitKey(33)
				if key == 27:
					break
				elif key == 32:
					image_path = 'opencv'+str(i)+'.png' 
					cv2.imwrite(image_path, frame)
					#image = Image.open('opencv'+str(i)+'.png')
					results=arabicocr.arabic_ocr(image_path,'out.jpg')
					print("..............")
					print(results)

					for i in range(len(results)):	
						word=results[i][1]
						word = word.strip().split()
						for word1 in word:
							if word1 in pre_words:						
								ind=pre_words.index(word1)
								playsound('sounds/'+str(ind)+'.mp3')
								break

						if i == len(results)-1:
							playsound('commands/r_fail.mp3')
						

					os.remove(image_path)

			else:
				break
		cv2.destroyAllWindows()
	

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
	def draw_labels(boxes, confs, colors, class_ids, classes, img, b_img): 
		indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
		font = cv2.FONT_HERSHEY_PLAIN
		coord_list=[]
		label_list=[]
		area_list = []
		for i in range(len(boxes)):
			if i in indexes:
				x, y, w, h = boxes[i]
				label = str(classes[class_ids[i]])
				color = colors[i]
				coord_list.append(boxes[i])
				label_list.append(str(classes[class_ids[i]]))
				area_list.append((w-x)*(h-y))
				cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
				cv2.putText(img, label, (x, y - 5), font, 1, color, 1)

		if len(label_list) > 3:
			ind = np.int64(np.argpartition(area_list, -3)[-3:])
			#print(ind)
			#print(type(ind))
			label_list = np.asarray(label_list)[ind.astype(int)]
			coord_list = np.asarray(coord_list)[ind.astype(int)]


		pos=[]
		for i in range(len(label_list)):
			x, y, w, h = coord_list[i]
			c_x = int(x+ w/2)
			c_y = int(y+ h/2)
			pos1 = "on the left" if c_x < 320 else "on the right"
			pos.append(pos1)
			cv2.circle(b_img,(c_x,c_y),10,(255,255,255), -1)
			cv2.putText(b_img, label_list[i]+pos1, (c_x, c_y- 10), font, 1,(255,255,255), 1)
		cv2.imshow("positions", b_img)
		cv2.imshow("Image", img)
		FULL=np.hstack(((cv2.cvtColor(b_img,cv2.COLOR_GRAY2BGR)),img))
		cv2.imshow("full",FULL)
		return label_list,pos
	
	

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
	
	@staticmethod
	def run2(obj):
		word, line = obj
		model = load_model()
    	# For each word in the image
		char_imgs = segment(line, word)
		txt_word = ''
    	# For each character in the word
		for char_img in char_imgs: 
			try:
				ready_char = prepare_char(char_img)
			except:
            	# breakpoint()
				continue
		feature_vector = featurizer(ready_char)
		predicted_char = model.predict([feature_vector])[0]
		txt_word += predicted_char
		return txt_word
	
	@staticmethod
	def run(image_path):
    	# Read test image
		full_image = cv2.imread(image_path)
		predicted_text = ''
		# Start Timer
		before = time.time()
		words = extract_words(full_image)       # [ (word, its line),(word, its line),..  ]
		pool = mp.Pool(mp.cpu_count())
		predicted_words = pool.map(words)
		pool.close()
		pool.join()
    	# Stop Timer
		after = time.time()

    # append in the total string.
		for word in predicted_words:
			predicted_text += word
			predicted_text += ' '
		exc_time = after-before
    # Create file with the same name of the image
    #img_name = image_path.split('\\')[1].split('.')[0]

    #with open(f'output/text/{img_name}.txt', 'w', encoding='utf8') as fo:
    #    fo.writelines(predicted_text)

    #predicted_text_corr = corr.contextual_correct(predicted_text)
		print(predicted_text)
		test = gTTS(text=predicted_text,lang='ar')


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
	Y=S_G()
	#video_play = args.play_video
	#image = args.image
	if webcam:
		#if args.verbose:
		#	print('---- Starting Web Cam object detection ----')
		Y.yolo_detect()
		#Y.imagetospeech()
		#Y.Test_Camera()
		#Y.register(2)
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