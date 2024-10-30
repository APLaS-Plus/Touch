from ultralytics import YOLO

model = YOLO("models/yolo11n.pt")

# Train the model
results = model.train(data=".//config//puzzleBlockDatasets.yaml", cfg=".//config//trainCfg.yaml")