from DisassemblyStateDetector import DisassemblyStateDetector
from OperatorComm import OperatorComm

import mediapipe as mp
import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time


MODEL_PATH = "best.pt"

SAVE_ROOT = "records"
os.makedirs(SAVE_ROOT, exist_ok=True)

session_name = time.strftime("session_%Y%m%d_%H%M%S")
session_dir = os.path.join(SAVE_ROOT, session_name)
os.makedirs(session_dir, exist_ok=True)


disassembly_detector = DisassemblyStateDetector(MODEL_PATH)

mp_hands = mp.solutions.hands

hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

operator_comm = OperatorComm(hands_detector, None)


pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipeline.start(config)

align = rs.align(rs.stream.color)


WC = cv2.VideoCapture(1)

WC.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
WC.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


EXPECTED_CONFIG = {
    "corps_verin": 1,
    "tige_verin": 1,
    "ressort": 1,
    "bouchon": 1
}


print("ESC ou q : Quitter | s : Capture")


try:

    while True:

        frames = pipeline.wait_for_frames()
        frames = align.process(frames)

        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame_rs = np.asanyarray(color_frame.get_data())

        state, counts = disassembly_detector.process(frame_rs)

        cv2.putText(
            frame_rs,
            f"Etat Verin: {state}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        y_offset = 80

        for piece_type, detected_count in counts.items():

            expected_count = EXPECTED_CONFIG.get(piece_type, 0)

            if detected_count == expected_count:
                status = "Correct"
                color = (0, 255, 0)
            elif detected_count == 0:
                status = "Manquant"
                color = (0, 0, 255)
            else:
                status = "Erreur Nb"
                color = (0, 165, 255)

            text = f"{piece_type} : {detected_count} ({status})"

            cv2.putText(
                frame_rs,
                text,
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

            y_offset += 30


        RET, F = WC.read()
        if not RET:
            continue

        rgb_usb = cv2.cvtColor(F, cv2.COLOR_BGR2RGB)

        results = hands_detector.process(rgb_usb)
        gesture = operator_comm.process(F, results)

        if gesture:
            cv2.putText(
                F,
                f"Geste: {gesture}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )


        cv2.imshow(
            "RealSense — Etat Verin",
            frame_rs
        )

        cv2.imshow(
            "Webcam Operateur — Gestes",
            F
        )

        KEY = cv2.waitKey(1)

        if KEY == 27 or KEY == ord("q"):
            break


finally:

    pipeline.stop()
    WC.release()

    cv2.destroyAllWindows()
    hands_detector.close()

    print("Arret propre du systeme")
