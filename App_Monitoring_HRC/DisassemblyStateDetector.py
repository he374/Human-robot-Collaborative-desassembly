from ultralytics import YOLO
from collections import Counter


class DisassemblyStateDetector:

    def __init__(self, model_path):

        self.model = YOLO(model_path)

        self.valid_classes = {
            "piece1",
            "piece2",
            "piece3",
            "piece4",
            "piece5",
            "piece6"
        }

        self.current_state = "INCONNU"

    def detect_parts(self, frame):

        results = self.model(frame, conf=0.3)

        detected_classes = []

        for r in results:

            if r.boxes is None:
                continue

            for box in r.boxes:

                cls_id = int(box.cls)
                class_name = self.model.names[cls_id]

                if class_name in self.valid_classes:
                    detected_classes.append(class_name)

        counts = Counter(detected_classes)

        return counts

    def determine_state(self, counts):

        p1 = counts.get("piece1", 0)
        p2 = counts.get("piece2", 0)
        p3 = counts.get("piece3", 0)
        p4 = counts.get("piece4", 0)
        p5 = counts.get("piece5", 0)
        p6 = counts.get("piece6", 0)

        total_detected = sum(counts.values())

        if total_detected == 0:
            return "VERIN_COMPLET"

        if p6 >= 2 and p5 == 0:
            return "ETAPE_1"

        if p6 >= 2 and p5 >= 1 and p5 < 2:
            return "ETAPE_2"

        if p6 >= 2 and p5 >= 2 and p3 == 0:
            return "ETAPE_3"

        if p6 >= 2 and p5 >= 2 and p3 >= 1 and p2 == 0:
            return "ETAPE_4"

        if p6 >= 2 and p5 >= 2 and p3 >= 1 and p2 >= 1 and p4 == 0:
            return "ETAPE_5"

        if p6 >= 2 and p5 >= 2 and p3 >= 1 and p2 >= 1 and p4 >= 1 and p1 == 0:
            return "ETAPE_6"

        if (
            p6 >= 2 and
            p5 >= 2 and
            p3 >= 1 and
            p2 >= 1 and
            p4 >= 1 and
            p1 >= 1
        ):
            return "ETAPE_7_FINAL"

        return "INCONNU"

    def process(self, frame):

        counts = self.detect_parts(frame)
        self.current_state = self.determine_state(counts)

        return self.current_state, counts
