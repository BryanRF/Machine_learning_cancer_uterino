import numpy as np
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from tensorflow import keras
from keras.utils import to_categorical
from ..models import TipoImagen
import os

class CNN:
    def __init__(self):
        self.model = self.load_data()

    def load_data(self, img_width=150, img_height=150, batch_size=32, validation_split=0.2):
        train_datagen = ImageDataGenerator(rescale=1.0/255, validation_split=validation_split)
        data_dir = 'C:/BrayanHTEC/Machine_learning_cancer_uterino/media'
        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical',
            subset='training')
        
        validation_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation')
        
        return train_generator, validation_generator

    def create_model(self, input_shape=(150, 150, 3)):
        slugs = TipoImagen.objects.values_list('slug', flat=True)
        num_classes = len(slugs)
     
        model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
        return model


    def train_model(self, model, train_generator, validation_generator, epochs=20):
        model.fit(train_generator, validation_data=validation_generator, steps_per_epoch=len(train_generator), validation_steps=len(validation_generator), epochs=epochs)

    def save_model(self, model, filename):
        model.save(filename)

    def predict_cancer(self,model, image_path, img_width=150, img_height=150):
        img = keras.preprocessing.image.load_img(
            image_path, target_size=(img_width, img_height)
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        predictions = model.predict(img_array)
        class_index = np.argmax(predictions[0])
        return class_index
    
    def load_model(self, model_filename):
            loaded_model = keras.models.load_model(model_filename)
            return loaded_model