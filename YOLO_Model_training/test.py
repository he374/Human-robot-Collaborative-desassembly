from ultralytics import YOLO

model = YOLO("runs_seg/pieces_seg4/weights/best.pt")

results = model("test2.jpg", conf=0.4)

results[0].show()
