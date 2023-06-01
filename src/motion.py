import cv2
import numpy as np

class MotionDetector:
    def __init__(self):
        self.prev_gray = None

    def detect_motion(self, frame):
      
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return False, frame


        frame_diff = cv2.absdiff(self.prev_gray, gray)
        _, frame_diff_thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)

       
        frame_diff_thresh = cv2.medianBlur(frame_diff_thresh, 5)
        frame_diff_thresh = cv2.dilate(frame_diff_thresh, None, iterations=3)

       
        contours, _ = cv2.findContours(frame_diff_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = False

        for contour in contours:
            if cv2.contourArea(contour) > 500:
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                motion_detected = True

    
        self.prev_gray = gray

        return motion_detected, frame
