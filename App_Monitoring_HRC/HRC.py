import tkinter as tk
from tkinter import ttk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import threading
import time
import cv2
import pyrealsense2 as rs
import numpy as np

from PIL import Image, ImageTk

from DisassemblyStateDetector import DisassemblyStateDetector
from OperatorComm import OperatorComm

import mediapipe as mp


class HRC:

    def __init__(self, tasks):

        self.tasks = []
        self._parse_tasks(tasks)

        self.state_history = []
        self.current_state = None
        self.waiting_validation = False

        self.paused = False
        self.stopped = False

        self.running = True
        self.pipeline = None
        self.webcam = None

        self.root = tk.Tk()
        self.root.title("HRC Monitoring System")

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_monitoring = tk.Frame(self.notebook)
        self.notebook.add(self.tab_monitoring, text="Monitoring")

        main_frame = tk.Frame(self.tab_monitoring)
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.gantt_frame = tk.Frame(main_frame)
        self.gantt_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.camera_frame = tk.Frame(main_frame)
        self.camera_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

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

        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.gantt_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._update_chart()

        self.tab_execution = tk.Frame(self.notebook)
        self.notebook.add(self.tab_execution, text="Execution Temps Réel")

        exec_main_frame = tk.Frame(self.tab_execution)
        exec_main_frame.pack(fill=tk.BOTH, expand=True)

        exec_left = tk.Frame(exec_main_frame)
        exec_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.state_label = tk.Label(
            exec_left,
            text="Etat courant : ...",
            font=("Arial", 14, "bold")
        )
        self.state_label.pack(pady=10)

        self.validation_label = tk.Label(
            exec_left,
            text="Validation opérateur : ...",
            font=("Arial", 12)
        )
        self.validation_label.pack(pady=5)

        self.gesture_label = tk.Label(
            exec_left,
            text="Geste détecté : ...",
            font=("Arial", 12, "bold"),
            fg="blue"
        )
        self.gesture_label.pack(pady=5)

        self.progress = ttk.Progressbar(
            exec_left,
            orient="horizontal",
            length=400,
            mode="determinate"
        )
        self.progress.pack(pady=10)

        self.fig2, self.ax2 = plt.subplots(figsize=(8, 4))
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=exec_left)
        self.canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        exec_right = tk.Frame(exec_main_frame)
        exec_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.exec_webcam_label = tk.Label(exec_right)
        self.exec_webcam_label.pack(pady=5)

        self.exec_rs_label = tk.Label(exec_right)
        self.exec_rs_label.pack(pady=5)

        MODEL_PATH = "best.pt"
        self.disassembly_detector = DisassemblyStateDetector(MODEL_PATH)

        mp_hands = mp.solutions.hands
        self.hands_detector = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        self.operator_comm = OperatorComm(self.hands_detector, None)

        self.camera_thread = threading.Thread(
            target=self._start_cameras,
            daemon=True
        )
        self.camera_thread.start()

        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _parse_tasks(self, tasks):
        for name, start, end in tasks:
            if end >= start:
                self.tasks.append((name, float(start), float(end)))

    def _update_chart(self):

        self.ax.clear()

        for i, (name, start, end) in enumerate(self.tasks):
            self.ax.barh(i, end - start, left=start)

        self.ax.set_yticks(range(len(self.tasks)))
        self.ax.set_yticklabels([t[0] for t in self.tasks])
        self.ax.set_title("Gantt Théorique")

        self.canvas.draw()

    def _update_realtime_gantt(self):

        self.ax2.clear()

        for i, (state, duration) in enumerate(self.state_history):
            self.ax2.barh(i, duration)

        self.ax2.set_yticks(range(len(self.state_history)))
        self.ax2.set_yticklabels([s[0] for s in self.state_history])
        self.ax2.set_title("Gantt Temps Réel")

        self.canvas2.draw()

    def _start_cameras(self):

        cameras_ok = True

        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.pipeline.start(config)
            self.align = rs.align(rs.stream.color)
        except:
            cameras_ok = False
            self.pipeline = None

        self.webcam = cv2.VideoCapture(1)

        if not self.webcam.isOpened():
            cameras_ok = False

        if not cameras_ok:
            self.root.after(
                0,
                lambda: self.camera_status_label.config(
                    text="Caméras non connectées",
                    fg="red"
                )
            )
            return
        else:
            self.root.after(
                0,
                lambda: self.camera_status_label.config(
                    text="Caméras connectées",
                    fg="green"
                )
            )

        while self.running:

            try:
                frames = self.pipeline.wait_for_frames()
                frames = self.align.process(frames)
                color_frame = frames.get_color_frame()

                if color_frame:

                    frame_rs = np.asanyarray(color_frame.get_data())
                    state, _ = self.disassembly_detector.process(frame_rs)

                    if state != self.current_state and not self.waiting_validation:

                        self.current_state = state
                        self.waiting_validation = True
                        self.paused = False
                        self.stopped = False

                        self.root.after(
                            0,
                            lambda s=state:
                            self.state_label.config(text=f"Etat courant : {s}")
                        )

                        self.root.after(
                            0,
                            lambda:
                            self.validation_label.config(
                                text="Validation opérateur requise"
                            )
                        )

                        threading.Thread(
                            target=self._validate_state,
                            args=(state,),
                            daemon=True
                        ).start()

                    frame_rs_rgb = cv2.cvtColor(frame_rs, cv2.COLOR_BGR2RGB)
                    img2 = Image.fromarray(frame_rs_rgb)
                    imgtk2 = ImageTk.PhotoImage(img2)

                    self.rs_label.imgtk = imgtk2
                    self.rs_label.configure(image=imgtk2)

                    self.exec_rs_label.imgtk = imgtk2
                    self.exec_rs_label.configure(image=imgtk2)

                ret, frame_usb = self.webcam.read()

                if ret:
                    frame_usb_rgb = cv2.cvtColor(frame_usb, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_usb_rgb)
                    imgtk = ImageTk.PhotoImage(img)

                    self.webcam_label.imgtk = imgtk
                    self.webcam_label.configure(image=imgtk)

                    self.exec_webcam_label.imgtk = imgtk
                    self.exec_webcam_label.configure(image=imgtk)

            except:
                break

        self._release_cameras()

    def _validate_state(self, state):

        start_time = time.time()
        pause_start = None
        total_pause = 0

        self.progress["value"] = 0

        while self.running and self.waiting_validation:

            ret, frame = self.webcam.read()
            if not ret:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands_detector.process(rgb)

            gesture = self.operator_comm.process(frame, results)

            self.root.after(
                0,
                lambda g=gesture:
                self.gesture_label.config(
                    text=f"Geste détecté : {g}"
                )
            )

            if gesture == "START" and self.paused:
                self.paused = False
                total_pause += time.time() - pause_start

                self.root.after(
                    0,
                    lambda:
                    self.validation_label.config(
                        text="Reprise opérateur "
                    )
                )

            elif gesture == "PAUSE" and not self.paused:
                self.paused = True
                pause_start = time.time()

                self.root.after(
                    0,
                    lambda:
                    self.validation_label.config(
                        text="Validation en pause ⏸"
                    )
                )

            elif gesture == "STOP":
                self.stopped = True
                self.waiting_validation = False

                self.root.after(
                    0,
                    lambda:
                    self.validation_label.config(
                        text="Tâche arrêtée "
                    )
                )
                break

            if self.paused:
                continue

            elapsed = time.time() - start_time - total_pause
            progress = min(elapsed * 20, 100)

            self.root.after(
                0,
                lambda v=progress:
                self.progress.config(value=v)
            )
            if gesture == "VALIDATE":

                duration = elapsed
                self.state_history.append((state, duration))

                self.root.after(0, self._update_realtime_gantt)

                self.root.after(
                    0,
                    lambda:
                    self.validation_label.config(
                        text="Etat validé "
                    )
                )

                self.waiting_validation = False
                break


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

    app = HRC(ta)
    app.run()
