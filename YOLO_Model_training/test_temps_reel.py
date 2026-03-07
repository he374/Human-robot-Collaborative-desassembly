import pyrealsense2 as rs
import numpy as np
import cv2
import os
from datetime import datetime
from ultralytics import YOLO


model_path = "runs_seg/pieces_seg4/weights/best.pt"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Modèle introuvable : {model_path}")

model = YOLO(model_path)
print(" v Modèle chargé")


record_dir = "recordings"
os.makedirs(record_dir, exist_ok=True)

# Nom vidéo avec date
video_name = datetime.now().strftime(
    "record_%Y%m%d_%H%M%S.avi"
)
video_path = os.path.join(record_dir, video_name)


pipeline = rs.pipeline()
config = rs.config()

WIDTH, HEIGHT, FPS = 640, 480, 30

config.enable_stream(
    rs.stream.color,
    WIDTH, HEIGHT,
    rs.format.bgr8,
    FPS
)

pipeline.start(config)

print("Camera RealSense lancée")


fourcc = cv2.VideoWriter_fourcc(*"XVID")

video_writer = cv2.VideoWriter(
    video_path,
    fourcc,
    FPS,
    (WIDTH, HEIGHT)
)

print(f"Enregistrement vidéo : {video_path}")


try:
    while True:

        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        frame = np.asanyarray(
            color_frame.get_data()
        )

        results = model(frame, conf=0.4)

        detected_classes = set()

        for r in results:

            boxes = r.boxes

            if boxes is None:
                continue

            for box in boxes:

                x1, y1, x2, y2 = map(
                    int, box.xyxy[0]
                )

                cls_id = int(box.cls)
                class_name = model.names[cls_id]
                conf = float(box.conf)

                detected_classes.add(class_name)

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                label = f"{class_name} {conf:.2f}"

                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        if detected_classes:
            print(
                "Classes détectées :",
                detected_classes
            )

        video_writer.write(frame)

        cv2.imshow(
            "Detection temps reel - RealSense",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


finally:
    pipeline.stop()
    video_writer.release()
    cv2.destroyAllWindows()

    print("Enregistrement terminé")
