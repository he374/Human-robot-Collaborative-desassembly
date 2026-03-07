import os

base_path = r"C:\plp detection verin"

train_dir = os.path.join(base_path, "images", "train")
val_dir   = os.path.join(base_path, "images", "val")

with open(os.path.join(base_path, "train.txt"), "w") as f:
    for img in os.listdir(train_dir):
        f.write(f"images/train/{img}\n")

with open(os.path.join(base_path, "val.txt"), "w") as f:
    for img in os.listdir(val_dir):
        f.write(f"images/val/{img}\n")

print("Fichiers train.txt et val.txt créés ")
