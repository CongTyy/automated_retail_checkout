from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("AIC/ori_fake_eval/best.pt")  # load a pretrained model (recommended for training)

# Use the model
# model.train(data="coco128.yaml", epochs=3)  # train the model
metric = model.val(data="aic.yaml", imgsz = 1280,  save=True, save_txt=True, conf=0.5)
print(metric)