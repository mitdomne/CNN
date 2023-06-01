from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import os

def get_train_test_data_generators(directory, target_size=(128, 128), batch_size=32, color_mode='rgb'):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        directory + '/training_set',
        target_size=target_size,
        batch_size=batch_size,
        color_mode=color_mode,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        directory + '/test_set',
        target_size=target_size,
        batch_size=batch_size,
        color_mode=color_mode,
        class_mode='categorical'
    )

#     return train_generator, test_generator
# def get_train_test_data_generators(directory, target_size=(128, 128), batch_size=32, color_mode='rgb'):
#     train_datagen = ImageDataGenerator(
#         rescale=1./255,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True,
#         rotation_range=20,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         brightness_range=[0.2,1.0]
#     )

#     test_datagen = ImageDataGenerator(rescale=1./255)

#     train_generator = train_datagen.flow_from_directory(
#         directory + '/training_set',
#         target_size=target_size,
#         batch_size=batch_size,
#         color_mode=color_mode,
#         class_mode='categorical'
#     )

#     test_generator = test_datagen.flow_from_directory(
#         directory + '/test_set',
#         target_size=target_size,
#         batch_size=batch_size,
#         color_mode=color_mode,
#         class_mode='categorical'
#     )

    return train_generator, test_generator

def build_model(num_classes):
    pretrained_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    for layer in pretrained_model.layers[:-4]:
        layer.trainable = False

    model = Sequential()
    model.add(pretrained_model)

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5)) 
    model.add(Dense(num_classes, activation='softmax'))

    return model

def train_model(model, train_generator, test_generator, num_epochs=20):#test với 50 9 classes thì bị overfitting, 40 là rơi vào 0.01-1.00 ok 
    model.compile(loss='categorical_crossentropy',                      #tăng 200 ảnh -chỉnh epoch xuống 30 tránh overfitting
                  optimizer=Adam(learning_rate=0.0001),  
                  metrics=['accuracy'])

    checkpoint_path = os.path.join(model_directory, 'model_checkpoint.h5')
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True)

    model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=num_epochs,
        validation_data=test_generator,
        validation_steps=len(test_generator),
        callbacks=[checkpoint]
    )
    return model

def get_num_classes(directory):
    train_generator = ImageDataGenerator().flow_from_directory(
        directory + '/training_set',
        class_mode='categorical'
    )
    num_classes = train_generator.num_classes
    return num_classes

if __name__ == '__main__':
    data_directory = './dataset'
    model_directory = './model'
 
    input_size = (128, 128)
    num_classes = get_num_classes(data_directory)

    train_generator, test_generator = get_train_test_data_generators(data_directory, target_size=input_size)

    model = build_model(num_classes)

    trained_model = train_model(model, train_generator, test_generator, num_epochs=30)

    checkpoint_path = os.path.join(model_directory, 'model_checkpoint.h5')
    trained_model.save(checkpoint_path)

    model_path = os.path.join(model_directory, 'my_model.h5')
    trained_model.save(model_path)
#---------------------Final-----------------------------------------------------
# from keras.applications import VGG16
# from keras.models import Sequential
# from keras.layers import Flatten, Dense, Dropout
# from keras.optimizers import Adam
# from keras.preprocessing.image import ImageDataGenerator
# from keras.callbacks import ModelCheckpoint
# import os

# def get_train_test_data_generators(directory, target_size=(128, 128), batch_size=32, color_mode='rgb'):
#     train_datagen = ImageDataGenerator(
#         rescale=1./255,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True
#     )

#     test_datagen = ImageDataGenerator(rescale=1./255)

#     train_generator = train_datagen.flow_from_directory(
#         directory + '/training_set',
#         target_size=target_size,
#         batch_size=batch_size,
#         color_mode=color_mode,
#         class_mode='categorical'
#     )

#     test_generator = test_datagen.flow_from_directory(
#         directory + '/test_set',
#         target_size=target_size,
#         batch_size=batch_size,
#         color_mode=color_mode,
#         class_mode='categorical'
#     )

#     return train_generator, test_generator

# def build_model(num_classes):
#     pretrained_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
#     for layer in pretrained_model.layers[:-4]:
#         layer.trainable = False

#     model = Sequential()
#     model.add(pretrained_model)

#     model.add(Flatten())
#     model.add(Dense(256, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(num_classes, activation='softmax'))

#     return model

# def train_model(model, train_generator, test_generator, num_epochs=20):
#     model.compile(loss='categorical_crossentropy',
#                   optimizer=Adam(learning_rate=0.0001),
#                   metrics=['accuracy'])

#     checkpoint_path = './model/model_checkpoint.h5'
#     checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True)

#     model.fit(
#         train_generator,
#         steps_per_epoch=len(train_generator),
#         epochs=num_epochs,
#         validation_data=test_generator,
#         validation_steps=len(test_generator),
#         callbacks=[checkpoint]
#     )
#     return model

# def get_num_classes(directory):
#     train_generator = ImageDataGenerator().flow_from_directory(
#         directory + '/training_set',
#         class_mode='categorical'
#     )
#     num_classes = train_generator.num_classes
#     return num_classes

# if __name__ == '__main__':
#     data_directory = './dataset'
 
#     input_size = (128, 128)
#     num_classes = get_num_classes(data_directory)

#     train_generator, test_generator = get_train_test_data_generators(data_directory, target_size=input_size)

#     model = build_model(num_classes)

#     trained_model = train_model(model, train_generator, test_generator, num_epochs=6)

#     model.save('./model/my_model.h5')
