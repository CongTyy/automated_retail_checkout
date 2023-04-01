from pandas_ods_reader import read_ods
from calendar import c
import json
import cv2
import os
from matplotlib.pyplot import flag
import numpy as np
import base64
import labelme
from math import atan2
import random
def convertScale(img, alpha, beta):
    new_img = img * alpha + beta
    new_img[new_img < 0] = 0
    new_img[new_img > 255] = 255
    return new_img.astype(np.uint8)

def automatic_brightness_and_contrast(image, clip_hist_percent=25):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)

    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    auto_result = convertScale(image, alpha=alpha, beta=beta)
    return (auto_result, alpha, beta)
'''
def increase_brightness(image):
	hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
	hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
	out_img = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

	kernel = np.array([[1,4,6,4,1], [4,16,24,16,4], [6,24,-476,24,6], [4,16,24,16,4], [1,4,6,4,1]]) * (-1 / 256)

	image = cv2.filter2D(image, -1, kernel)

	auto_result, alpha, beta = automatic_brightness_and_contrast(image)

	return auto_result
'''

def increase_brightness(img):
# 	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# 	h, s, v = cv2.split(hsv)

# 	lim = 255 - value
# 	v[v <= lim] += value
# 	v[v > 230] = 200	
# 	v[v <= 170] += 25
# 	final_hsv = cv2.merge((h, s, v))
# 	img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
	hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
	hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
	img = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
	#for c in range(0, 3):
	#	img[:,:,c] = cv2.equalizeHist(img[:,:,c])
	return img
sheet_index = 1
df = read_ods("rawdata/class116.ods" , sheet_index )
new_label_list = np.zeros(116, dtype=int)
sample_3_classes = np.zeros(3, dtype=int)
box = []
jar = []
bag = []
for b, j, bg in zip(df['Box'], df['Jar'], df['Bag']):
	if not np.isnan(b):
		new_label_list[int(b-1)] = 0
		box.append(int(b-1))
	if not np.isnan(j):
		new_label_list[int(j-1)] = 1 
		jar.append(int(j-1))
	if not np.isnan(bg):
		new_label_list[int(bg-1)] = 2
		bag.append(int(bg-1))

# new_label_list = np.zeros(116, dtype=int)
# sample_3_classes = np.zeros(3, dtype=int)
# box = []
# jar = []
# bag = [] 
# for i in df['ID']:
# 	if df['Box'][i-1] is not None:
# 		new_label_list[int(i-1)] += 0
# 		box.append(int(i-1))
# 	elif df['jar'][i-1] is not None:
# 		new_label_list[int(i-1)] += 1
# 		jar.append(int(i-1))
# 	elif df['bag'][i-1] is not None:
# 		new_label_list[int(i-1)] += 2
# 		bag.append(int(i-1))

def argsort(seq):
		return sorted(range(len(seq)), key=seq.__getitem__)

def rotational_sort(list_of_xy_coords, centre_of_rotation_xy_coord, clockwise=True):
	cx,cy=centre_of_rotation_xy_coord
	angles = [atan2(x-cx, y-cy) for x,y in list_of_xy_coords]
	indices = argsort(angles)
	if clockwise:
		return [list_of_xy_coords[i] for i in indices]
	else:
		return [list_of_xy_coords[i] for i in indices[::-1]]
def _drawBoundingBox(cnt):
	x,y,w,h = cv2.boundingRect(cnt)
	# img = cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),2)
	return x,y,w,h

classes = ["Advil Liquid Gel","Advil Tablets","Alcohol Pads","Aleve PM","Aleve Tablets","All Free and Clear Pods","Arm and Hammer Toothpaste","Bandaid Variety","Banza","Barilla Penne","Barilla Red Lentil","Barilla Rotini Pasta","Barnums Animal Crackers","Bayer Chewable 3 Pack","Bayer Lower Dose","Benadryl Allergy Tablets","Biore","Brown Rice","Bumble Bee Tuna","Cane Sugar","Chamomile Tea","Chicken Bouillon","Children_s Allegra Allergy","Children_s Zyrtec","Chocolate Fudge Poptart","Chocolate Pocky","Chocolate Strawberries","Claratin Chewable","Coffee Mate Creamer","Crackers and Cheese","Cream Cheese","Crest White Strips","Dove Cool Moisturizer","Dove Soap Sensitive","Dove Body Wash","Downy Odor Defense","Dreft Scent Beads","Dry Eye Relief","Excedrine Migraine","Extra Spearmint","Flonase","Flow Flex Covid 19 Test","French Roast Starbucks","Frosting Mix Chocolate","Fudge Oreos","Funfetti Cake","Gain Fireworks Scent Beads","Gain Flings","Garnier Shampoo","Glad Drawstring Trash Bag","Gluten Free Brownies","Gluten Free Pasta","Gournay Cheese","Healthy Choice Chicken Alfredo","Heavy Duty Forks","Hefty Small Trash Bags","Hello Toothpaste","Honey Maid Graham Crackers","Ice Cream Sandwiches","Ice Breakers Ice Cubes","Jello Vanilla Pudding","Kerrygold Butter","Kleenex","Lipton Noodle Soup","Mac 'n' Cheese Shells","Macaroni and Cheese Original","Matzo Balls","McCormick Natures Food Coloring","Milk Duds","Minute Tapioca","Miss Dior","Mixed Berries Hibiscus Tea","M&Ms Peanuts","Mochi Ice Cream","Motrin Ib Migraine","Mr Clean","Nasacort Allergy","Nasal Decongestant","Nasal Strip","Nature Valley Granola Bars","Nintendo Switch Controllers","Onion Soup Dip Mix","Pedialyte Powder","Peets Keurig","Peet's Coffee Grounds Dark Roast","Playtex Sport","Pocky Strawberry","Pork Sausage","Pure Vanilla Extract","Raisinets","Ranch Seasoning","Reese's Pieces","Rewetting Drops Contact Solution","Smores Poptarts","Snuggle Dryer Sheets","Sour Patch Kids","Stevia in the Raw","Strawberry Jello","Stress Relief Spray All Natural","Sunmaid Raisins","Swiss Miss Hot Chocolate","Tampax","Tide Pods","Toothpicks","Tostada Shells","Total Home Scent Boos","Tussin","Tylenol Arthritis","Unstoppables","Vapo Rub","Vick's Pure Zzz's","Vinyl Gloves","Visine Red Eye Hydrating Comfort","Whoppers","Wild Strawberry Tea","Woolite Delicates _Attempt_"]
# def click_and_crop(event, x, y, flags, param):
# 	global refPt, cropping,flag_init
# 	if flag_init == 0:
# 		if event == cv2.EVENT_LBUTTONDOWN:
# 			refPt = [(x, y)]
# 			cropping = True
# 		elif event == cv2.EVENT_LBUTTONUP:
# 			refPt.append((x, y))
# 			cropping = False
# 			cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
# 			cv2.imshow("image", image)
# 			flag_init = 1

# Sửa từ chỗ này
# -----	data.py
# 	|--	background
# 	|--	image_storage
			# |--	1
			# 	|--	00001_1.jpg
			# 	|--	00001_117.jpg
			# 	|-- ..
			# |--	2
			# 	|-- 00002_2.jpg
			# 	|-- 00002_118.jpg 
			# 	|-- ..
			# |-- ..
#	|-- segmentation_labels
background_path = "./rawdata/bg/" # Đường dẫn chứa ảnh background
output_path = "./rawdata/val/" # Thư mục chứa ảnh đã tạo
img_path = "./rawdata/data_split/" # Thư mục chứa ảnh đã phân các class
seg_path = "./rawdata/segmentation_labels/" # Thư mục seg lấy luôn của BTC lúc chưa chia ra từng class
index_sample = 0 # Đánh số ảnh - nếu tạo thêm thì chỉnh lại
total_sample_box = 1500 #Tổng số mẫu cho box
total_sample_jar = 1500
total_sample_bag = 1500
#--------------------------------------------#
# image = cv2.imread(os.path.join(background_path+os.listdir(background_path)[0]))
# clone = image.copy()
# cv2.namedWindow("image")
# cv2.setMouseCallback("image", click_and_crop)
# flag_init = 0
# while True:
# 	cv2.imshow("image", image)
# 	key = cv2.waitKey(1) & 0xFF
# 	if key == ord("r"):
# 		image = clone.copy()
# 		flag_init = 0
# 	elif key == 13:
# 		break
# if len(refPt) == 2:
# 	roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
# 	cv2.imshow("ROI", roi)
# 	cv2.waitKey(0)
# cv2.destroyAllWindows()
refPt = [(261, 225), (1734, 1048)]


if os.path.exists(output_path) == False:
	os.mkdir(output_path)
sample_flag = 0
sample_check_balance = np.zeros(116)
class_check = []
for i in range(0,116):
	class_check.append([])
max_box_class_sample = round(total_sample_box / len(box))
max_jar_class_sample = round(total_sample_jar / len(jar))
max_bag_class_sample = round(total_sample_bag / len(bag))
sample_over_list = []
sample_bg_over = []
flag_final = 0
# while(sum(sample_check_balance) < sample_sum-3 and sample_flag >= 0):
while(flag_final==0):
	print(sample_3_classes,end='\r')
	index_sample += 1
	f_yolo = open(output_path+str(index_sample)+".txt",'w',encoding="UTF-8")
	new_filename = str(index_sample) + ".jpg"
	background_list = os.listdir(background_path)
	
	bg_filename = random.choice([x for x in background_list if x not in sample_bg_over])
	sample_bg_over.append(bg_filename)
	if len(sample_bg_over) == len(background_list):
		sample_bg_over = []


	background = cv2.imread(os.path.join(background_path+bg_filename))
	background_copy = background.copy()
	background_mask = np.zeros([background.shape[0],background.shape[1]])
	x_bg = refPt[0][0]

	result = {
		"version": "5.0.1",
		"flags": {},
		"shapes": [],
		"imagePath": new_filename,
		"imageData": "",
		"imageHeight": background.shape[0],
		"imageWidth": background.shape[1]
	}
	
	for idx in range(random.randrange(1,5)):
	# for i in range(10):
		sample_over_check_list = [x for x in range(1, 117) if x not in sample_over_list]
		if len(sample_over_check_list) > 0:
			index_random = random.choice(sample_over_check_list)
			f1 = sample_3_classes[new_label_list[index_random-1]-1]
			f2 = sample_3_classes[new_label_list[index_random-1]]
			if new_label_list[index_random-1] + 1 > len(sample_3_classes) - 1:
				f3 = sample_3_classes[0]
			else:
				f3 = sample_3_classes[new_label_list[index_random-1]+1]
			
			while f2 + 1 - f1 > 10 or f2 + 1 - f3 > 10 and len(sample_over_check_list) > 4:					
				index_random = random.choice(sample_over_check_list)
				f1 = sample_3_classes[new_label_list[index_random-1]-1]
				f2 = sample_3_classes[new_label_list[index_random-1]]
				if new_label_list[index_random-1] + 1 > len(sample_3_classes) - 1:
					f3 = sample_3_classes[0]
				else:
					f3 = sample_3_classes[new_label_list[index_random-1]+1]
			src = os.listdir(img_path + str(index_random))
			filename_list = [x for x in src if x not in class_check[index_random-1]]
			if len(filename_list) == 0 :
				class_check[index_random-1] = []
				filename_list = [x for x in src if x not in class_check[index_random-1]]
			filename = random.choice(filename_list)
			class_check[index_random-1].append(filename)
			sample_check_balance[index_random-1] += 1
				
			
			label_name_1 = filename.split(".")[0].split("_")[0]
			img_original = cv2.imread(os.path.join(img_path + str(int(label_name_1)) + "/" + filename))
			img = increase_brightness(img_original)
			
			label_name = new_label_list[int(label_name_1)-1]		
			sample_3_classes[label_name] += 1
			label = cv2.imread(os.path.join(seg_path + filename.split(".")[0]+ "_seg" +  ".jpg"))
			
			label_gray = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
			(thresh, mask) = cv2.threshold(label_gray, 110, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
			mask = cv2.threshold(label_gray, thresh, 255, cv2.THRESH_BINARY)[1]
			
			if background.shape[0] - mask.shape[0] <= 0:
				y_bg = 0
			else:
				if background.shape[0] - mask.shape[0] > refPt[0][1]:
					y_bg = random.randrange(refPt[0][1], background.shape[0] - mask.shape[0])
				else:
					y_bg = refPt[0][1]
			if x_bg <= refPt[1][0] and y_bg <= refPt[1][1] and y_bg+mask.shape[0] <= background_mask.shape[0] and x_bg+mask.shape[1] <= background_mask.shape[1]:
				background_mask_copy = background_mask.copy()
				background_mask[y_bg:y_bg+mask.shape[0],x_bg:x_bg+mask.shape[1]] = mask
				condition = np.stack((mask,) * 3, axis=-1)
				abcxyz = background_copy[y_bg:y_bg+mask.shape[0],x_bg:x_bg+mask.shape[1]]

				img_none_background = np.where(condition,img,abcxyz)

				background_copy[y_bg:y_bg+mask.shape[0],x_bg:x_bg+mask.shape[1]] = img_none_background
				background_mask = cv2.bitwise_or(background_mask,background_mask_copy)
				# cv2.imshow("background_mask",background_mask)
				# cv2.imshow("background_copy",background_copy)
				
				background_ob = np.zeros([background.shape[0],background.shape[1]])
				background_ob[y_bg:y_bg+mask.shape[0],x_bg:x_bg+mask.shape[1]] = mask
				# if mask.shape[1] > 50:
				# 	x_bg += random.randrange(50,mask.shape[1])
				# else:
				# 	x_bg = 50
				x_bg += random.randrange(int(round(mask.shape[1]/4)),mask.shape[1])
				background_ob = cv2.normalize(src=background_ob, dst=None, alpha=10, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
				ret,thresh = cv2.threshold(background_ob,110,255,0) # nhị phân hóa bức ảnh bằng cách đặt ngưỡng, với giá trị của ngưỡng là 127 8im2, 
				contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) # tìm contour
				font = cv2.FONT_HERSHEY_COMPLEX
				points = []

				
				for cnt in contours :		
					approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
					n = approx.ravel() 
					i = 0

					for j in n :
						if(i % 2 == 0):
							x = n[i]
							y = n[i + 1]
							string = str(x) + " " + str(y) 
							xy = []
							xy.append(float(x))
							xy.append(float(y))
							points.append(xy)
						i = i + 1
				xf,yf=zip(*points)
				
				
				x_center=(max(xf)+min(xf))/2.
				y_center=(max(yf)+min(yf))/2.

				ob_inf = {
				"label": classes[int(label_name_1)-1],
				"points": rotational_sort(points, (x_center,y_center),True),
				"group_id": None,
				"shape_type": "polygon",
				"flags": {}
				}
				result["shapes"].append(ob_inf)
				
				
				
				condition = np.stack((background_mask,) * 3, axis=-1)
				img_new_3 = np.where(condition,background_copy,background)
				img_new = cv2.normalize(src=img_new_3, dst=None, alpha=10, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
				background_copy = img_new.copy()
				area_cnt = [cv2.contourArea(cnt) for cnt in contours]
				area_sort = np.argsort(area_cnt)[::-1]
				imgOrigin = img_new.copy()
				cnt = contours[area_sort[0]]
				x,y,w,h = _drawBoundingBox(cnt)

				x1 = x
				y1 = y
				x2 = x + w
				y2 = y + h

				height_img = background.shape[0]
				width_img = background.shape[1]

				x_center = (x2-x1)/2 + x1
				x_center = x_center/width_img
				x_center = round(x_center,6)

				y_center = (y2-y1)/2 + y1
				y_center = y_center/height_img
				y_center = round(y_center,6)

				W_label = (x2-x1)
				W_label = W_label/width_img
				W_label = round(W_label,6)

				H_label = (y2-y1)
				H_label = H_label/height_img
				H_label = round(H_label,6)
				f_yolo.write(str(int(label_name))+" "+str(x_center)+" "+str(y_center)+" "+str(W_label)+" "+str(H_label)+"\n")
				if index_random-1 in box:
					if sample_check_balance[index_random-1] >= max_box_class_sample:
						sample_over_list.append(index_random)
				elif index_random-1 in jar:
					if sample_check_balance[index_random-1] >= max_jar_class_sample:
						sample_over_list.append(index_random)
				elif index_random-1 in bag:
					if sample_check_balance[index_random-1] >= max_bag_class_sample:
						sample_over_list.append(index_random)
				if sum(sample_check_balance[i] for i in box) >= total_sample_box:
					for i in box:						
						sample_over_list.append(i+1)
				if sum(sample_check_balance[i] for i in jar) >= total_sample_jar:
					for i in jar:
						sample_over_list.append(i+1)
				if sum(sample_check_balance[i] for i in bag) >= total_sample_bag:
					for i in bag:
						sample_over_list.append(i+1)

			else:
				sample_3_classes[label_name] -= 1
				sample_check_balance[index_random-1] -= 1
		else:
			if sum(sample_check_balance[i] for i in box) < total_sample_box:
				for i in box:
					sample_over_list.remove(i+1)
			if sum(sample_check_balance[i] for i in jar) < total_sample_jar:
				for i in jar:
					sample_over_list.remove(i+1)
			if sum(sample_check_balance[i] for i in bag) < total_sample_bag:
				for i in bag:
					sample_over_list.remove(i+1)
			if sum(sample_check_balance) >= (total_sample_box+total_sample_jar+total_sample_bag):
				flag_final = 1
			break

	if sum(sample_check_balance) > sample_flag: sample_flag = sum(sample_check_balance)
	else: sample_flag = -1
	if sample_flag != -1: 
		cv2.imwrite(os.path.join(output_path + new_filename),img_new)
		data_img = labelme.LabelFile.load_image_file(output_path + new_filename)
		image_data = base64.b64encode(data_img).decode('utf-8')
		result["imageData"] = image_data
	else:
		try:
			os.rmdir(output_path+str(index_sample)+".txt")
		except:
			pass

	# with open(output_path +new_filename.split('.')[0] +'.json', 'w', encoding='utf-8') as f:
	# 	json.dump(result, f, ensure_ascii=False, indent=2)
	f_yolo.close()
print(sample_3_classes)
