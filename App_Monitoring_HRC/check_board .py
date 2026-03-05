import cv2
import numpy as np
import pyrealsense2 as rs
import os
import time


class CheckerboardCapture:
    def __init__(self,
                 squares=(8, 8),
                 resolution=(1280, 720),
                 fps=30,
                 out_dir="captures_checkerboard"):

        self.SQUARES = squares
        self.PATTERN_SIZE = (squares[0] - 1, squares[1] - 1)

        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.shot_idx = 0

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(
            rs.stream.color,
            resolution[0],
            resolution[1],
            rs.format.bgr8,
            fps
        )

        self.pipeline.start(self.config)

        print("Damier attendu:", self.SQUARES,
              "carrés ->", self.PATTERN_SIZE, "coins intérieurs")
        print("Touches:")
        print("  s : sauvegarder (si damier détecté)")
        print("  q : quitter")

    def detect_checkerboard(self, gray):

        gray_eq = cv2.equalizeHist(gray)

        flags_sb = (cv2.CALIB_CB_NORMALIZE_IMAGE |
                    cv2.CALIB_CB_EXHAUSTIVE |
                    cv2.CALIB_CB_ACCURACY)

        ok, corners = cv2.findChessboardCornersSB(
            gray_eq,
            self.PATTERN_SIZE,
            flags=flags_sb
        )
        
        
        if not ok:
            flags_std = (cv2.CALIB_CB_ADAPTIVE_THRESH |
                         cv2.CALIB_CB_NORMALIZE_IMAGE)

            ok, corners = cv2.findChessboardCorners(
                gray_eq,
                self.PATTERN_SIZE,
                flags=flags_std
            )

            if ok:
                criteria = (
                    cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER,
                    50,
                    1e-3
                )

                cv2.cornerSubPix(
                    gray_eq,
                    corners,
                    (11, 11),
                    (-1, -1),
                    criteria
                )

        return ok, corners

    def save_images(self, img, vis):

        ts = time.strftime("%Y%m%d-%H%M%S")
        base = f"cb_{ts}_{self.shot_idx:03d}"

        raw_path = os.path.join(self.out_dir, base + "_raw.png")
        ann_path = os.path.join(self.out_dir, base + "_annotated.png")

        cv2.imwrite(raw_path, img)
        cv2.imwrite(ann_path, vis)

        print("Saved:", raw_path)
        print("Saved:", ann_path)

        self.shot_idx += 1

    def run(self):

        try:
            while True:

                frames = self.pipeline.wait_for_frames()
                color = frames.get_color_frame()
                if not color:
                    continue

                img = np.asanyarray(color.get_data())
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                ok, corners = self.detect_checkerboard(gray)

                vis = img.copy()

                if ok:
                    cv2.drawChessboardCorners(
                        vis,
                        self.PATTERN_SIZE,
                        corners,
                        ok
                    )

                    cv2.putText(
                        vis,
                        f"OK pattern={self.PATTERN_SIZE} (press 's')",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
                else:
                    cv2.putText(
                        vis,
                        f"NOT FOUND pattern={self.PATTERN_SIZE}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2
                    )

                cv2.imshow("Checkerboard test", vis)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break

                if key == ord('s'):
                    if not ok:
                        print("Damier NON détecté -> pas de sauvegarde.")
                        continue

                    self.save_images(img, vis)

        finally:
            self.stop()

    def stop(self):
        self.pipeline.stop()
        cv2.destroyAllWindows()


"""if __name__ == "__main__":

    app = CheckerboardCapture(
        squares=(8, 8),
        resolution=(1280, 720),
        fps=30
    )

    app.run()"""
