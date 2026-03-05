import mediapipe as mp
import time

class OperatorComm:

    def __init__(self, hands_detector, depth_frame):
        self.hands_detector = hands_detector
        self.depth_frame = depth_frame
        self.stop_gesture_start = None
        self.last_beep_time = 0.0
        self.current_gesture = None

    def finger_up(self, lm, tip, mcp):
        return lm[tip].y < lm[mcp].y

    def finger_down(self, lm, tip, mcp):
        return lm[tip].y > lm[mcp].y

    def detect_gesture(self, hand_landmarks):

        mp_hands = mp.solutions.hands
        lm = hand_landmarks.landmark

        thumb_up = self.finger_up(
            lm,
            mp_hands.HandLandmark.THUMB_TIP,
            mp_hands.HandLandmark.THUMB_MCP
        )

        index = self.finger_up(
            lm,
            mp_hands.HandLandmark.INDEX_FINGER_TIP,
            mp_hands.HandLandmark.INDEX_FINGER_MCP
        )

        middle = self.finger_up(
            lm,
            mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            mp_hands.HandLandmark.MIDDLE_FINGER_MCP
        )

        pinky = self.finger_up(
            lm,
            mp_hands.HandLandmark.PINKY_TIP,
            mp_hands.HandLandmark.PINKY_MCP
        )

        # Mapping gestes
        if thumb_up and not index and not middle:
            return "VALIDATION"

        if index and middle and not pinky:
            return "START"

        if thumb_up and index and pinky:
            return "PAUSE"

        if not thumb_up and not index and not middle:
            return "STOP"

        return None

    def process(self, frame, results):

        if not results.multi_hand_landmarks:
            return None

        h, w, _ = frame.shape

        for hand in results.multi_hand_landmarks:

            xs = [lm.x for lm in hand.landmark]
            ys = [lm.y for lm in hand.landmark]

            x1, y1 = int(min(xs) * w), int(min(ys) * h)
            x2, y2 = int(max(xs) * w), int(max(ys) * h)

            gesture = self.detect_gesture(hand)
            self.current_gesture = gesture

            if gesture == "STOP":

                if self.stop_gesture_start is None:
                    self.stop_gesture_start = time.time()

                elapsed = time.time() - self.stop_gesture_start

                if elapsed >= 5.0:
                    return "STOP_CONFIRMED"

            else:
                self.stop_gesture_start = None

        return self.current_gesture
