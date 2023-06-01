#-----------Đọc từ file video có sẵn test.mp4------------
import cv2
import os
from datetime import datetime


video_path = './dataset/test_motion/test.mp4'


motion_images_dir = './dataset/data_motion'


if not os.path.exists(motion_images_dir):
    os.makedirs(motion_images_dir)


cap = cv2.VideoCapture(video_path)


if not cap.isOpened():
    print("Không thể mở video.")
    exit()


ret, prev_frame = cap.read()
ret, curr_frame = cap.read()


if not ret:
    print("Không thể đọc các khung hình từ video.")
    exit()


motion_detector = cv2.createBackgroundSubtractorMOG2()

while True:
   
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    mask = motion_detector.apply(curr_gray)

    mask = cv2.medianBlur(mask, 5)
    mask = cv2.dilate(mask, None, iterations=3)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 2500: #giao động tầm 1000-2000 cũng ok
            (x, y, w, h) = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h

   
            if 0.5 < aspect_ratio < 2.0:
            
                cv2.rectangle(curr_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                motion_image = curr_frame[y:y+h, x:x+w]
                now = datetime.now()
                img_name = now.strftime("%Y%m%d") + '-' + now.strftime("%H%M%S") + '.jpg'
                cv2.imwrite(os.path.join(motion_images_dir, img_name), motion_image)


    cv2.imshow("Motion Detection", curr_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    prev_frame = curr_frame
    ret, curr_frame = cap.read()

cap.release()
cv2.destroyAllWindows()

#-----------Đọc trực tiếp từ camera----------------
# import cv2
# import os
# from datetime import datetime

# motion_images_dir = './dataset/data_motion'


# if not os.path.exists(motion_images_dir):
#     os.makedirs(motion_images_dir)


# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("Không thể mở video từ camera.")
#     exit()


# ret, prev_frame = cap.read()
# ret, curr_frame = cap.read()

# if not ret:
#     print("Không thể đọc các khung hình từ video.")
#     exit()


# motion_detector = cv2.createBackgroundSubtractorMOG2()

# while True:
  
#     prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
#     curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

  
#     mask = motion_detector.apply(curr_gray)


#     mask = cv2.medianBlur(mask, 5)
#     mask = cv2.dilate(mask, None, iterations=3)


#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


#     for contour in contours:
#         if cv2.contourArea(contour) > 2000: #giao động tầm 1000-2000 cũng ok
#             (x, y, w, h) = cv2.boundingRect(contour)
#             aspect_ratio = float(w) / h

#             # Kiểm tra hình dạng của đối tượng
#             if 0.5 < aspect_ratio < 2.0:
#                 # Vẽ hình chữ nhật xung quanh đối tượng
#                 cv2.rectangle(curr_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#                 motion_image = curr_frame[y:y+h, x:x+w]
#                 now = datetime.now()
#                 img_name = now.strftime("%Y%m%d") + '-' + now.strftime("%H%M%S") + '.jpg'
#                 cv2.imwrite(os.path.join(motion_images_dir, img_name), motion_image)

  
#     cv2.imshow("Motion Detection", curr_frame)

  
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

   
#     prev_frame = curr_frame
#     ret, curr_frame = cap.read()

# cap.release()
# cv2.destroyAllWindows()
