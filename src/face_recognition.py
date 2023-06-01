# import os
# import cv2
# import numpy as np
# from keras.models import load_model


# model = load_model('model/my_model.h5')

# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# video_capture = cv2.VideoCapture(0)


# train_dir = './dataset/training_set'

# class_names = os.listdir(train_dir)
# class_names = sorted(class_names)

# print(class_names)

# while True:

#     ret, frame = video_capture.read()

#     color_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

 
#     faces = face_cascade.detectMultiScale(color_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))


#     for (x, y, w, h) in faces:
        
#         face_image_color = frame[y:y+h, x:x+w]
#         face_image_color = cv2.resize(face_image_color, (128, 128))
#         face_image_color = np.expand_dims(face_image_color, axis=0)
#         face_image_color = face_image_color / 255.0

  
#         predictions = model.predict(face_image_color)
#         predicted_class_index = np.argmax(predictions[0])
#         confidence = np.max(predictions)
      
#         confidence_percent = round(confidence * 100)

#         if confidence > 0.8:
#             if predicted_class_index < len(class_names):
#                predicted_class_name = class_names[predicted_class_index]
#             else:
#                 predicted_class_name = "Unknown"
#         else:
#             predicted_class_name = "Unknown"

#         text = f"{predicted_class_name} - {confidence_percent}%"

#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#         cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


#     cv2.imshow('Face Recognition', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# video_capture.release()
# cv2.destroyAllWindows()
#---------------------Đọc từ video ------------
import os
import cv2
import numpy as np
from keras.models import load_model


model = load_model('model/my_model.h5')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture('./dataset/test_face/Obama_Biden.mp4')

train_dir = './dataset/training_set'

class_names = os.listdir(train_dir)
class_names = sorted(class_names)

print(class_names)

while True:

    ret, frame = video_capture.read()

    if not ret:
        break

    color_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = face_cascade.detectMultiScale(color_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        
        face_image_color = frame[y:y+h, x:x+w]
        face_image_color = cv2.resize(face_image_color, (128, 128))
        face_image_color = np.expand_dims(face_image_color, axis=0)
        face_image_color = face_image_color / 255.0

        predictions = model.predict(face_image_color)
        predicted_class_index = np.argmax(predictions[0])
        confidence = np.max(predictions)
      
        confidence_percent = round(confidence * 100)

        if confidence > 0.8:
            if predicted_class_index < len(class_names):
               predicted_class_name = class_names[predicted_class_index]
            else:
                predicted_class_name = "Unknown"
        else:
            predicted_class_name = "Unknown"

        text = f"{predicted_class_name} - {confidence_percent}%"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Face-Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
