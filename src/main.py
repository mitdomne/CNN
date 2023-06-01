import cv2
import os
from datetime import datetime
from face_recognition import FaceRecognizer
from motion import MotionDetector
import threading

train_dir = './dataset/training_set'


#face_recognizer = FaceRecognizer(train_dir)


motion_detector = MotionDetector()

motion_images_dir = './dataset/data_motion'

if not os.path.exists(motion_images_dir):
    os.makedirs(motion_images_dir)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Không thể mở video từ camera.")
    exit()

#def face_recognition_thread():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc khung hình từ camera.")
            break
        frame_copy = frame.copy()
        face_locations, face_names = face_recognizer.detect_faces(frame_copy)
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def motion_detection_thread():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc khung hình từ camera.")
            break
        motion_detected, motion_frame = motion_detector.detect_motion(frame)
        if motion_detected:
            now = datetime.now()
            img_name = now.strftime("%Y%m%d") + '-' + now.strftime("%H%M%S") + '.jpg'
            cv2.imwrite(os.path.join(motion_images_dir, img_name), motion_frame)
        cv2.imshow('Motion Detection', motion_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Tạo các luồng riêng biệt cho nhận dạng gương mặt và phát hiện chuyển động
#face_thread = threading.Thread(target=face_recognition_thread)
motion_thread = threading.Thread(target=motion_detection_thread)


#face_thread.start()
motion_thread.start()

#face_thread.join()
motion_thread.join()

cap.release()
cv2.destroyAllWindows()
