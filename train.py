from ultralytics import YOLO

# model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8m.pt")  # load a pretrained model (recommended for training
model.train(data="aic.yaml", 
            epochs = 500, 
            batch = 8, 
            imgsz = 1280, 
            device = 'cpu', 
            project='AIC', 
            optimizer = 'Adam',
            box = 7.5,
            cls = 0.5,
            dfl = 1.5,
            flipud = 0.5,
            fliplr = 0.5,
            mosaic = 0.0,
            degrees = 5,
            patience = 20,
            )  # train the model