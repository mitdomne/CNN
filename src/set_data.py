# import cv2
# import os
# import numpy as np
# from PIL import Image
# from sklearn.model_selection import train_test_split

# def preprocess_image(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#     if len(faces) == 0:
#         return None, None
#     elif len(faces) > 1:
#         return None, faces
#     (x, y, w, h) = faces[0]
#     face_image = gray[y:y+h, x:x+w]
#     resized_image = cv2.resize(face_image, (256, 256))
#     return resized_image, (x, y, w, h)

# def draw_rectangle(image, coordinates):
#     (x, y, w, h) = coordinates
#     cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# def capture_and_preprocess_images(class_name, num_images=20):
#     images = []
#     video_capture = cv2.VideoCapture(0)
#     print("Nhấn phím 'c' để chụp ảnh...")
#     while True:
#         ret, frame = video_capture.read()
#         preprocessed_image, face_coordinates = preprocess_image(frame)
#         if preprocessed_image is not None:
#             draw_rectangle(frame, face_coordinates)
#         cv2.imshow('Chup anh', frame)
#         if cv2.waitKey(1) & 0xFF == ord('c'):
#             if face_coordinates is not None:
#                 break
#             else:
#                 print("Cảnh báo: Có nhiều hơn một khuôn mặt trong khung hình. Vui lòng chỉ để lại một khuôn mặt trong khung hình.")
#     count = 0
#     while count < num_images:
#         ret, frame = video_capture.read()
#         preprocessed_image, _ = preprocess_image(frame)
#         if preprocessed_image is not None:
#             cv2.imshow('Chup anh', frame)
#             print(f'Đã chụp ảnh {count+1}/{num_images}')
#             images.append(preprocessed_image)
#             count += 1
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     video_capture.release()
#     cv2.destroyAllWindows()
#     return images

# if __name__ == '__main__':
#     output_dir = './dataset'
#     class_name = input('Nhập tên người chụp: ')
#     num_images = 150
#     images = capture_and_preprocess_images(class_name, num_images)

#     train_images, test_images = train_test_split(images, test_size=0.2, random_state=42)

#     train_dir = os.path.join(output_dir, 'training_set', class_name)
#     if not os.path.exists(train_dir):
#         os.makedirs(train_dir)
#     for i, img in enumerate(train_images):
#         image_path = os.path.join(train_dir, f'image_{i}.jpg')
#         Image.fromarray(img).save(image_path)

#     test_dir = os.path.join(output_dir, 'test_set', class_name)
#     if not os.path.exists(test_dir):
#         os.makedirs(test_dir)
#     for i, img in enumerate(test_images):
#         image_path = os.path.join(test_dir, f'image_{i}.jpg')
#         Image.fromarray(img).save(image_path)
#----------------------------Lấy từ video -------------------------
import cv2
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None, None
    elif len(faces) > 1:
        return None, faces
    (x, y, w, h) = faces[0]
    face_image = gray[y:y+h, x:x+w]
    resized_image = cv2.resize(face_image, (256, 256))
    return resized_image, (x, y, w, h)

def draw_rectangle(image, coordinates):
    (x, y, w, h) = coordinates
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

def capture_and_preprocess_images(class_name, num_images=20, video_file="my_video.mp4"):
    images = []
    video_capture = cv2.VideoCapture(video_file)
    count = 0
    while count < num_images:
        ret, frame = video_capture.read()
        if not ret:
            break
        preprocessed_image, face_coordinates = preprocess_image(frame)
        if preprocessed_image is not None:
            draw_rectangle(frame, face_coordinates)
            cv2.imshow('Chup anh', frame)
            print(f'Đã chụp ảnh {count+1}/{num_images}')
            images.append(preprocessed_image)
            count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()
    return images

if __name__ == '__main__':
    output_dir = './dataset'
    class_name = input('Nhập tên người chụp: ')
    num_images = 200
    video_file = './dataset/test_face/Joe_Biden.mp4' 
    images = capture_and_preprocess_images(class_name, num_images, video_file)

    train_images, test_images = train_test_split(images, test_size=0.2, random_state=42)

    train_dir = os.path.join(output_dir, 'training_set', class_name)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    for i, img in enumerate(train_images):
        image_path = os.path.join(train_dir, f'image_{i}.jpg')
        Image.fromarray(img).save(image_path)

    test_dir = os.path.join(output_dir, 'test_set', class_name)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    for i, img in enumerate(test_images):
        image_path = os.path.join(test_dir, f'image_{i}.jpg')
        Image.fromarray(img).save(image_path)
