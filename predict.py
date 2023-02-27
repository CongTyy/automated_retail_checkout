from ultralytics import YOLO
from PIL import Image
import cv2

model = YOLO("AIC/train12/weights/best.pt")  # load a pretrained model (recommended for training)
# print(model)
# exit()
# from ndarray
# im2 = cv2.imread("datasets/test/6.png")
results = model.predict(source="rawdataset/testA/testA_1.mp4", conf = 0.5, device = 'cuda:0', save=True, save_txt=True)  # save predictions as labels


# print(results)

# define a video capture object
# vid = cv2.VideoCapture("rawdataset/testA/testA_1.mp4")

# frame_width = int(vid.get(3))
# frame_height = int(vid.get(4))
# out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))#(1080, 1920, 3)

# while(True):
	
# 	# Capture the video frame
# 	# by frame
# 	ret, frame = vid.read()
	
# 	out.write(frame)


# 	if cv2.waitKey(1) & 0xFF == ord('q'):
# 		break

# # After the loop release the cap object
# vid.release()
# # Destroy all the windows
# cv2.destroyAllWindows()
