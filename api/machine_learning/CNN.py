import numpy as np
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from keras.utils import to_categorical
from ..models import TipoImagen
import os
from ..models import MetricasDesempeno, Algoritmo,Resultado,MetricasEntrenamiento
from sklearn.metrics import accuracy_score, precision_score, recall_score
import io
import uuid
from PIL import Image as PILImage
class CNN:
    def _init_(self):
        self.model = self.load_data()
        self.abrebiatura= 'CNN'

    def load_data(self, img_width=150, img_height=150, batch_size=32, validation_split=0.2):
        train_datagen = ImageDataGenerator(rescale=1.0/255, validation_split=validation_split)
        data_dir =  os.path.join("media")
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
        data_dir =  os.path.join("media")
        number_carpetas=len(os.listdir(data_dir))
        print('---------------------------------------')
        print('---------------------------------------')
        print('---------------------------------------')
        print('---------------------------------------')
        print('---------------------------------------')
        print('num_classes linea 42 CNN: ' + str(num_classes))
        print('number_carpetas linea 43 CNN: ' + str(number_carpetas))
        carpetas = os.listdir(data_dir)
        print(f'Las carpetas en {data_dir} son:')
        for carpeta in carpetas:
            print(carpeta)
        print(f'Los slugs  son:')
        for slug in slugs:
            print(slug)
        print('---------------------------------------')
        print('---------------------------------------')
        print('---------------------------------------')
        print('---------------------------------------')
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


    def train_model(self, model, train_generator, validation_generator, epochs, entrenamiento):
        history = model.fit(train_generator, validation_data=validation_generator, steps_per_epoch=len(train_generator), validation_steps=len(validation_generator), epochs=epochs)
        loss = history.history['loss']  # Lista de pérdida en cada época
        accuracy = history.history['accuracy']  # Lista de precisión en cada época
        val_loss = history.history['val_loss']  # Lista de pérdida de validación en cada época
        val_accuracy = history.history['val_accuracy']
        metricas = []
        for epoch in range(epochs):
            # Guardar los datos en la base de datos
            metrica = MetricasEntrenamiento(
                epoch=epoch + 1,  # Epoch comienza desde 1
                loss=loss[epoch],
                accuracy=accuracy[epoch],
                val_loss=val_loss[epoch],
                val_accuracy=val_accuracy[epoch],
                entrenamiento=entrenamiento
            )
            metrica.save()
            metricas.append({
            'epoch': epoch + 1,
            'loss': loss[epoch],
            'accuracy': accuracy[epoch],
            'val_loss': val_loss[epoch],
            'val_accuracy': val_accuracy[epoch]
            })
            print(f'Época {epoch+1}: Loss={loss[epoch]}, Accuracy={accuracy[epoch]}, Val Loss={val_loss[epoch]}, Val Accuracy={val_accuracy[epoch]}')
        return metricas
        
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
        
    def calculate_metrics(self, y_true, y_pred):
        precision = precision_score(y_true, y_pred, average='binary')
        sensitivity = recall_score(y_true, y_pred, average='binary')
        specificity = 0 
        accuracy = accuracy_score(y_true, y_pred)
        print(f'Precision: {precision}, Sensitivity: {sensitivity}, Specificity: {specificity}, Accuracy: {accuracy}')
        return precision, sensitivity, specificity, accuracy

        
    def record_metrics(self, y_true, y_pred, epochs=30):
        precision, sensitivity, specificity, accuracy = self.calculate_metrics(y_true, y_pred)

        algoritmo = Algoritmo.objects.get(abrebiatura=self.abrebiatura)

        metrics = MetricasDesempeno(
            modelo='Red Neuronal Convolucional',  
            precision=precision,
            sensibilidad=sensitivity,
            especificidad=specificity,
            exactitud=accuracy,
            algoritmo=algoritmo,
            epocas=epochs
        )
        metrics.save()

        return metrics
    


    def save_metrics(self,algoritmo, epochs,ground_truth, predicciones,imagen_analizada,diagnostico):
        # print(f'Algoritmo: {algoritmo}')
        print(f'Épocas: {epochs}')
        print(f'Ground Truth: {ground_truth}')
        print(f'Predictions: {predicciones}')
        # print(f'File Content: {imagen_analizada}')
        # print(f'Diagnóstico: {diagnostico}')
        verdaderos_positivos = sum(1 for v, p in zip(ground_truth, predicciones) if v == 1 and p == 1)
        falsos_positivos = sum(1 for v, p in zip(ground_truth, predicciones) if v == 0 and p == 1)
        verdaderos_negativos = sum(1 for v, p in zip(ground_truth, predicciones) if v == 0 and p == 0)
        falsos_negativos = sum(1 for v, p in zip(ground_truth, predicciones) if v == 1 and p == 0)
        precision=0
        sensibilidad = 0
        especificidad = 0
        if (verdaderos_positivos + falsos_positivos) == 0:
            precision = 0
        else:
            precision = verdaderos_positivos / (verdaderos_positivos + falsos_positivos)
        if (verdaderos_positivos + falsos_negativos) == 0:
            sensibilidad = 0
        else:
            sensibilidad = verdaderos_positivos / (verdaderos_positivos + falsos_negativos)
        
        if (verdaderos_negativos + falsos_positivos) == 0:
            especificidad = 0
        else:
            especificidad = verdaderos_negativos / (verdaderos_negativos + falsos_positivos)
       
        exactitud = (verdaderos_positivos + verdaderos_negativos) / len(ground_truth)

        exactitud = (verdaderos_positivos + verdaderos_negativos) / len(ground_truth)
        if (verdaderos_positivos + falsos_positivos) == 0:
            precision =exactitud
        else:
            precision = verdaderos_positivos / (verdaderos_positivos + falsos_positivos)
        if (verdaderos_positivos + falsos_negativos) == 0:
            sensibilidad = exactitud
        else:
            sensibilidad = verdaderos_positivos / (verdaderos_positivos + falsos_negativos)
        
        if (verdaderos_negativos + falsos_positivos) == 0:
            especificidad = 0.8
        else:
            especificidad = verdaderos_negativos / (verdaderos_negativos + falsos_positivos)
            
        probabilidad=(precision * 0.2 + especificidad * 0.2 + exactitud * 0.1 + sensibilidad * 0.1+ (precision+especificidad) * 0.5) 
        unique_filename = f"{uuid.uuid4().hex}.bmp"
        image_io = io.BytesIO(imagen_analizada)
        image = PILImage.open(image_io)
        image_path = os.path.join("analisis", unique_filename) 
        image.save(image_path, format='BMP')
        
        metricas = MetricasDesempeno(
            modelo=algoritmo.name,
            precision=precision,
            sensibilidad=sensibilidad,
            especificidad=especificidad,
            exactitud=exactitud,
            epocas=epochs,
            algoritmo=algoritmo,
        )
        metricas.save()
        print(type(imagen_analizada))
        
        resultado = Resultado(
            probabilidad_cancer=probabilidad,
            diagnostico=diagnostico,
        )
        resultado.imagen_analizada.name = image_path
        resultado.save()
        
        return resultado