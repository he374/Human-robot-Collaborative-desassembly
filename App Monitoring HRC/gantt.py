import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import threading
import cv2
import pyrealsense2 as rs
import numpy as np

from PIL import Image, ImageTk

from DisassemblyStateDetector import DisassemblyStateDetector
from OperatorComm import OperatorComm

import mediapipe as mp


class Gantt:

    def __init__(self, tasks):

        self.tasks = []
        self._parse_tasks(tasks)


        self.root = tk.Tk()
        self.root.title(
            "Gantt des taches et affichage des videos"
        )

        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Gantt
        self.gantt_frame = tk.Frame(main_frame)
        self.gantt_frame.pack(
            side=tk.LEFT,
            fill=tk.BOTH,
            expand=True
        )

        # Cameras
        self.camera_frame = tk.Frame(main_frame)
        self.camera_frame.pack(
            side=tk.RIGHT,
            fill=tk.BOTH,
            expand=True
        )

        self.camera_status_label = tk.Label(
            self.camera_frame,
            text="Connexion des caméras...",
            fg="orange",
            font=("Arial", 12, "bold")
        )
        self.camera_status_label.pack(pady=5)

        self.webcam_label = tk.Label(self.camera_frame)
        self.webcam_label.pack(pady=5)

        self.rs_label = tk.Label(self.camera_frame)
        self.rs_label.pack(pady=5)


        self.fig, self.ax = plt.subplots(
            figsize=(8, 4)
        )

        self.canvas = FigureCanvasTkAgg(
            self.fig,
            master=self.gantt_frame
        )

        self.canvas.get_tk_widget().pack(
            fill=tk.BOTH,
            expand=True
        )

        self._update_chart()

        MODEL_PATH = "best.pt"

        self.disassembly_detector = \
            DisassemblyStateDetector(MODEL_PATH)

        mp_hands = mp.solutions.hands

        self.hands_detector = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        self.operator_comm = \
            OperatorComm(self.hands_detector, None)


        self.running = True

        self.camera_thread = threading.Thread(
            target=self._start_cameras,
            daemon=True
        )

        self.camera_thread.start()

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.close
        )

    def _parse_tasks(self, tasks):

        for name, start, end in tasks:

            if end >= start:

                self.tasks.append(
                    (name, float(start), float(end))
                )


    def _update_chart(self):

        self.ax.clear()

        for i, (name, start, end) \
                in enumerate(self.tasks):

            duration = end - start

            self.ax.barh(
                i,
                duration,
                left=start
            )

        self.ax.set_yticks(
            range(len(self.tasks))
        )

        self.ax.set_yticklabels(
            [task[0] for task in self.tasks]
        )

        self.ax.set_xlabel("Temps (s)")
        self.ax.set_title("Diagramme de Gantt")

        self.canvas.draw()


    def _start_cameras(self):

        cameras_ok = True

        # RealSense
        try:

            self.pipeline = rs.pipeline()
            config = rs.config()

            config.enable_stream(
                rs.stream.color,
                640, 480,
                rs.format.bgr8,
                30
            )

            config.enable_stream(
                rs.stream.depth,
                640, 480,
                rs.format.z16,
                30
            )

            self.pipeline.start(config)

            self.align = rs.align(
                rs.stream.color
            )

        except:

            cameras_ok = False
            self.pipeline = None

        # Webcam
        self.webcam = cv2.VideoCapture(1)

        self.webcam.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            640
        )

        self.webcam.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            480
        )

        if not self.webcam.isOpened():
            cameras_ok = False

        # Status UI
        if not cameras_ok:

            self.root.after(
                0,
                lambda:
                self.camera_status_label.config(
                    text="Caméras non connectées",
                    fg="red"
                )
            )
            return

        else:

            self.root.after(
                0,
                lambda:
                self.camera_status_label.config(
                    text="Caméras connectées",
                    fg="green"
                )
            )

        while self.running:

            try:

                # REALSENSE
                frames = \
                    self.pipeline.wait_for_frames()

                frames = \
                    self.align.process(frames)

                color_frame = \
                    frames.get_color_frame()

                if color_frame:

                    frame_rs = \
                        np.asanyarray(
                            color_frame.get_data()
                        )

                    # detection de l'état du vérin et comptage des pièces
                    state, counts = \
                        self.disassembly_detector.process(
                            frame_rs
                        )

                    # Etat global
                    cv2.putText(
                        frame_rs,
                        f"Etat Verin: {state}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2
                    )

                    # Comptage brut
                    y_offset = 80

                    for piece_type, detected_count \
                            in counts.items():

                        text = \
                            f"{piece_type} : {detected_count}"

                        cv2.putText(
                            frame_rs,
                            text,
                            (20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 0),
                            2
                        )

                        y_offset += 30

                    frame_rs = cv2.cvtColor(
                        frame_rs,
                        cv2.COLOR_BGR2RGB
                    )

                    img2 = Image.fromarray(frame_rs)
                    imgtk2 = ImageTk.PhotoImage(img2)

                    self.rs_label.imgtk = imgtk2
                    self.rs_label.configure(
                        image=imgtk2
                    )

                # webcam
                ret, frame_usb = \
                    self.webcam.read()

                if ret:

                    rgb_usb = cv2.cvtColor(
                        frame_usb,
                        cv2.COLOR_BGR2RGB
                    )

                    results = \
                        self.hands_detector.process(
                            rgb_usb
                        )

                    gesture = \
                        self.operator_comm.process(
                            frame_usb,
                            results
                        )

                    if gesture:

                        cv2.putText(
                            frame_usb,
                            f"Geste: {gesture}",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2
                        )

                    frame_usb = cv2.cvtColor(
                        frame_usb,
                        cv2.COLOR_BGR2RGB
                    )

                    img = Image.fromarray(frame_usb)
                    imgtk = ImageTk.PhotoImage(img)

                    self.webcam_label.imgtk = imgtk
                    self.webcam_label.configure(
                        image=imgtk
                    )

            except:
                break

        self._release_cameras()

    def _release_cameras(self):

        if self.pipeline:
            try:
                self.pipeline.stop()
            except:
                pass

        if self.webcam:
            try:
                self.webcam.release()
            except:
                pass

        try:
            self.hands_detector.close()
        except:
            pass


    def close(self):

        self.running = False
        self._release_cameras()
        self.root.destroy()


    def run(self):

        self.root.mainloop()



if __name__ == "__main__":

    from tasks import tasks

    t = tasks(None)
    ta = t.tasks

    app = Gantt(ta)
    app.run()
