from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8l.yaml")  # build a new model from scratch
model = YOLO("yolov8/multi_layers_wGAN_gp/train/weights/best.pt")  # load a pretrained model (recommended for training)

metric = model.val(data="aic_test.yaml", imgsz = 1280,  save=True, conf = 0.1)
print(metric)