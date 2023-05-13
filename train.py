from ultralytics import YOLO

# model = YOLO("best.pt")  # load a pretrained model (recommended for training
# model = YOLO("yolov8/multidomain_dcGAN_yoloN/ok/weights/best.pt")  # load a pretrained model (recommended for training
model = YOLO("yolov8n.pt")
model.train(data="aic.yaml", 
            epochs = 300, 
            batch = 8, 
            imgsz = 1280, 
            device = 'cuda:1', 
            project='yolov8/multi_layers_wGAN_gp', 
            optimizer = 'Adam',
            box = 7.5, #7.5,
            cls = 0.5, #0.5,
            dfl = 1.5,
            flipud = 0.5,
            fliplr = 0.5,
            mosaic = 0.5,
            degrees = 5,
            patience = 100,
            mixup = 0.5,
            save_period = 1,
            lr0 = 1e-4,
            lrf = 1e-4,
            warmup_epochs = 0,
            workers = 8,
            cache  = False
            )  # train the 