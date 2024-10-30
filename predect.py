from ultralytics import YOLO

# Load a model
model = YOLO(".//models//best.pt",task="detect")  # pretrained YOLO11n model

# import yaml
# y = yaml.safe_load(open("config//puzzleBlockDatasets.yaml"))
# print(y)

# import os
# print(os.getcwd())
# print(os.listdir(y["path"]))

print("Detecting...")

results = model(
    source="images", conf=0.5, iou=0.45, device="cpu", imgsz=(1920,1080), task="detect",save_txt=True,save_conf=True,save=True
)