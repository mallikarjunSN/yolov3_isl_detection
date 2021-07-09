import cv2
import numpy as np 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--webcam', help="True/False", default=False)
parser.add_argument('--play_video', help="True/False", default=False)
parser.add_argument('--image', help="True/False", default=False)
parser.add_argument('--video_path', help="Path of video file", default="videos/car_on_road.mp4")
parser.add_argument('--image_path', help="Path of image to detect objects", default="/images/busy_street.jpg")
parser.add_argument('--verbose', help="To print statements", default=True)
args = parser.parse_args()

#Load yolo
def load_yolo():
	net = cv2.dnn.readNet("yolov3_custom_last.weights", "yolov3_custom.cfg")
	classes = []
	with open("labels.txt", "r") as f:
		classes = [line.strip() for line in f.readlines()]

	# classes = ['A']

	layers_names = net.getLayerNames()
	output_layers = [layers_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
	colors = np.random.uniform(0, 255, size=(len(classes), 3))
	return net, classes, colors, output_layers

def load_image(img_path):
	# image loading
	img = cv2.imread(img_path)
	img = cv2.resize(img, None, fx=1.0, fy=1.0)
	height, width, channels = img.shape
	return img, height, width, channels

def start_webcam():
	cap = cv2.VideoCapture(0)

	return cap


def display_blob(blob):
	'''
		Three images each for RED, GREEN, BLUE channel
	'''
	# cv2.imshow ('screen', img)
	for b in blob:
		for n, imgb in enumerate(b):
			cv2.imshow(str(n), imgb)

def detect_objects(img, net, outputLayers):			
	blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
	net.setInput(blob)
	outputs = net.forward(outputLayers)
	return blob, outputs

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
			
text = ''


# def getMax(s:str):
# 	count = {}

# 	for ch in str:
# 		if ch in count:
# 			count[ch]+=1
# 		else:
# 			count[ch]=1

# 	print(max(count,key=count.get))
# 	return max(count,key=count.get)


count = 0
def draw_labels(boxes, confs, colors, class_ids, classes, img): 
	indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)

	if args.webcam:
		cv2.putText(img, "word obtained : {}".format(word), (30,50), cv2.FONT_HERSHEY_PLAIN, 1.5, [0,0,0], 2)

	font = cv2.FONT_HERSHEY_PLAIN
	global text
	for i in range(len(boxes)):
		if i in indexes:
			x, y, w, h = boxes[i]
			label = str(classes[class_ids[i]])
			# print(label)
			if label == "BKSP":
				text = text[:-1]
			else:
				text+=label
			getWord(text,img)
			# print(len(classes[class_ids[i]]))
			# print(confs)
			confs = [max(confs)]
			color = [0, 169, 255 ] #colors[i]
			# print(type(color))
			label="{} {:2.2f}%".format(label,max(confs)*100.0)
			cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
			cv2.putText(img, label, (x, y - 5), font, 2, color, 3)
	# cv2.namedWindow ("Image", cv2.WINDOW_AUTOSIZE)
	# cv2.resizeWindow("Image",500,500)
	# cv2.setWindowProperty ("Image", 10, 10)
	cv2.imshow("Image", img)

def image_detect(img_path): 
	model, classes, colors, output_layers = load_yolo()
	image, height, width, channels = load_image(img_path)
	blob, outputs = detect_objects(image, model, output_layers)
	boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
	draw_labels(boxes, confs, colors, class_ids, classes, image)
	while True:
		key = cv2.waitKey(1)
		if key == 27:
			break

def webcam_detect():
	model, classes, colors, output_layers = load_yolo()
	cap = start_webcam()
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

word = ''
def getWord(txt:str,img):
	global text
	txt = txt.replace("SPACE"," ")
	global word
	word = " "

	flag = False

	if "BKSP" in txt:
		flag = True
		txt = txt.replace("BKSP","")
	# 	return

	for ch in txt:
		if txt.count(ch) <3 :
			txt = txt.replace(ch,"")

	prev = ''
	occ = 0
	for ch in txt:
		if prev!= ch or occ>3:
			word += ch
			prev = ch
			occ=0
		else:
			occ+=1

	if args.webcam:
		print("WORD = "+ word)


if __name__ == '__main__':
	webcam = args.webcam
	video_play = args.play_video
	image = args.image or args.image_path != "/images/busy_street.jpg"
	if webcam:
		if args.verbose:
			print('---- Starting Web Cam object detection ----')
		webcam_detect()
	if video_play:
		video_path = args.video_path
		if args.verbose:
			print('Opening '+video_path+" .... ")
		start_video(video_path)
	if image:
		image_path = args.image_path
		if args.verbose:
			print("Opening "+image_path+" .... ")
		image_detect(image_path)

	
	cv2.destroyAllWindows()