from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')

model.train(data='/Users/codedpro/Documents/computer vision/Weather iC', epochs=20, imgsz=64)