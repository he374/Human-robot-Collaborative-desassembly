from ultralytics import YOLO
import torch
import os



DATA_YAML = "data.yaml"
MODEL = "yolov8n-seg.pt"

EPOCHS = 100
BATCH = 4
IMGSZ = 640

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Device utilisé :", DEVICE)

def count_labels(folder):
    total = 0
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.endswith(".txt"):
                total += 1
    return total

labels_train = count_labels("labels/train")
labels_val   = count_labels("labels/val")

print(f"Labels train trouvés : {labels_train}")
print(f"Labels val trouvés   : {labels_val}")

if labels_train == 0:
    print("Aucun label trouvé : Vérifie structure dossier")
    exit()

model = YOLO(MODEL)

model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    device=DEVICE,
    workers=2,
    project="runs_seg",
    name="pieces_seg",
    pretrained=True,
    patience=20,
    optimizer="Adam",
    verbose=True
)

print("Training finished")
