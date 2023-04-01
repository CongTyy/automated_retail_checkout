from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8l.yaml")  # build a new model from scratch
model = YOLO("yolov8/yolov8n/train6/weights/best.pt")  # load a pretrained model (recommended for training)

metric = model.val(data="aic.yaml", imgsz = 1280,  save=True, save_txt=True, conf=0.1)
print(metric)