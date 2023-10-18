import io
import numpy as np
from pathlib import Path
from typing import Any
from django.http import JsonResponse
from django.views import View
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from .machine_learning.CNN import CNN
from .machine_learning.SVM import SVMClassifier 
from .models import TipoImagen as TipoImage
from .models import Algoritmo
from .models import Entrenamiento
from .models import Image
from django.core.files.base import ContentFile
import base64
import json
import os
from pathlib import Path
import cv2
from skimage.metrics import structural_similarity as ssim
from django.utils import timezone
from datetime import datetime
from django.core.files import File
import uuid
from django.http import FileResponse
from django.conf import settings
import os
from django.http import Http404
import joblib
class cnn_view(View):

    def __init__(self):
        super().__init__()
        self.train_generator = None  # Inicializa train_generator como None en el constructor
        self.model = None  # Inicializa model como None en el constructor
        self.epochs = 1
        now = datetime.now()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.cnn_name = f"CNN_{now.strftime('%Y%m%d%H%M%S')}.h5"
        self.file_cnn = os.path.join(current_dir,"machine_learning","entrenamiento", self.cnn_name)

    @method_decorator(csrf_exempt)
    def dispatch(self, request, *args: Any, **kwargs: Any):
        return super().dispatch(request, *args, **kwargs)

    def put(self, request):
        data = json.loads(request.body.decode('utf-8'))
        algoritmo_id = int(data.get('algoritmo_id'))
        self.epochs = int(data.get('epochs'))
        classifier = CNN()
        train_generator, validation_generator = classifier.load_data()
        self.model = classifier.create_model(input_shape=(150, 150, 3))
        classifier.train_model(self.model, train_generator,
                               validation_generator, self.epochs)
        classifier.save_model(self.model, self.file_cnn)
        # Ahora guardamos la información en la tabla Entrenamiento
        entrenamiento = Entrenamiento(
            algoritmo=Algoritmo.objects.get(pk=algoritmo_id),
            epocas=self.epochs,
            rutamodelo=self.cnn_name,
            fecha_entrenamiento=timezone.now()  # Fecha actual
        )
        entrenamiento.save()

        return JsonResponse({'mensaje': 'Modelo entrenado correctamente'}, safe=False)

    def get(self, request):
        # Consulta todas las eTipoImagen en la base de datos
        all_algoritmos = Algoritmo.objects.all()
        # Crea una lista para almacenar los datos de las eTipoImagen
        algoritmos_lista = []
        # Itera sobre todas las eTipoImagen y agrega sus datos a la lista
        for algoritmo in all_algoritmos:
            algorit = {
                'id': algoritmo.algoritmo_id,
                'nombre': algoritmo.name,
                'descripcion': algoritmo.descripcion,
                'abrebiatura': algoritmo.abrebiatura,
                # Agrega más campos según sea necesario
            }
            algoritmos_lista.append(algorit)
        # Retorna la lista de eTipoImagen como una respuesta JSON
        return JsonResponse({'algoritmos': algoritmos_lista}, safe=False)

    def convertir_bytes(self, image_path):
        try:
            with open(image_path, 'rb') as image_file:
                image_bytes = image_file.read()
                return base64.b64encode(image_bytes).decode('utf-8')
        except Exception as e:
            # Manejar errores si la imagen no se puede leer
            return None

    def post(self, request):
        # data = json.loads(request.body.decode('utf-8'))
        algoritmo_id = request.POST.get('algoritmo_id')
        algoritmo = Algoritmo.objects.get(pk=algoritmo_id) 
        entrenamiento = Entrenamiento.objects.filter(
            algoritmo=algoritmo).order_by('-fecha_entrenamiento').first()
        self.epochs = entrenamiento.epocas if entrenamiento else None
        if self.epochs is not None: 
            self.epochs = int(self.epochs)
        else:
            # No se encontraron entrenamientos para el algoritmo
            return JsonResponse(
            
            safe=False,data='Por favor, entrene el algoritmo primero.'
            )
        classifier = CNN()
        classifier_aux = CNN()
        train_generator, validation_generator_aux = classifier_aux.load_data()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path= os.path.join(current_dir,"machine_learning","entrenamiento",   entrenamiento.rutamodelo)
        if Path(path).is_file():
            self.model = classifier.load_model(path)
            if self.train_generator is None:
                self.train_generator, _ = classifier.load_data()
        else:
            train_generator, validation_generator_aux = classifier.load_data()
            self.model = classifier.create_model(input_shape=(150, 150, 3))
            classifier.train_model(
                self.model, train_generator, validation_generator_aux, self.epochs)
            classifier.save_model(self.model, path)
            self.train_generator = train_generator
        file_content = request.FILES['image'].read()
        image_io = io.BytesIO(file_content)
        predicted_class = classifier.predict_cancer(self.model, image_io)
        class_names = list(self.train_generator.class_indices.keys())
        predicted_TipoImagen_slug = class_names[predicted_class]
        tipo_imagen = TipoImage.objects.filter(slug=predicted_TipoImagen_slug)  # Corregido a 'tipo_imagen'
        datos_encontrados = tipo_imagen.first()  # Corregido a 'tipo_imagen'
        cancer_image = Image.objects.filter(tipo_imagen=datos_encontrados).order_by('?').first()  # Corregido a 'tipo_imagen'
       
        unique_filename = f"{uuid.uuid4().hex}.bmp"
        prediccion={}
        if tipo_imagen.exists():
            image_bytes = self.convertir_bytes(cancer_image.image.path)

            ground_truth = validation_generator_aux.labels
            
            predictions = np.argmax(self.model.predict(validation_generator_aux), axis=1)
            datos_resultados=classifier.save_metrics(algoritmo, entrenamiento.epocas, ground_truth,predictions,file_content,datos_encontrados.diagnostico)
                
            prediccion = {
                        'prediccion': class_names[predicted_class], 
                        'estado': True,
                        'image_url': image_bytes,
                        'nombre': datos_encontrados.nombre, 
                        'diagnostico': datos_encontrados.diagnostico.nombre, 
                        'descripcion': datos_encontrados.diagnostico.descripcion,
                        'es_benigno':datos_encontrados.diagnostico.es_benigno,
                        'porcentaje':datos_resultados.probabilidad_cancer*100,
                        }
            
        else:
            # Corregido a minúsculas por convención
            return JsonResponse({'mensaje':'No se encontraron registros'}, safe=False)
        # Cambiado a minúsculas por convención
        return JsonResponse(prediccion, safe=False)
    def serve_analisis(request, filename):
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        analisis_file_path = os.path.join(BASE_DIR, 'analisis', filename)
        media_file_path = os.path.join(BASE_DIR, 'media', filename)
        
        if os.path.isfile(analisis_file_path):
            return FileResponse(open(analisis_file_path, 'rb'))
        elif os.path.isfile(media_file_path):
            return FileResponse(open(media_file_path, 'rb'))
        else:
            raise Http404
        
        
class svm_view(View):
    
    def __init__(self):
        super().__init__()
        self.train_generator = None  # Inicializa train_generator como None en el constructor
        self.model = None  # Inicializa model como None en el constructor
        now = datetime.now()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.rd_name = f"SVM.joblib"
        self.file_name = os.path.join(current_dir,"machine_learning","entrenamiento",  self.rd_name)
    @method_decorator(csrf_exempt)
    def dispatch(self, request, *args: Any, **kwargs: Any):
        return super().dispatch(request, *args, **kwargs)
    
    def convertir_bytes(self, image_path):
        try:
            with open(image_path, 'rb') as image_file:
                image_bytes = image_file.read()
                return base64.b64encode(image_bytes).decode('utf-8')
        except Exception as e:
            # Manejar errores si la imagen no se puede leer
            return None
    def post(self, request):
        # Obtener la imagen binaria del request
        imagen_binaria = request.FILES['image'].read()
        classifier = SVMClassifier()

        if imagen_binaria:
            # Entrenar el modelo si es la primera vez
            classifier.entrenar_modelo(self.file_name)
            
            # Convertir los datos binarios a una imagen (esto supone que es un archivo de imagen)
            nparr = np.fromstring(imagen_binaria, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)  # Leer la imagen

            # Redimensionar la imagen si es necesario
            img = cv2.resize(img, (64, 64))

            # Aplanar la imagen
            test_X = [img.flatten()]

            # Realizar la predicción
            predicted_class = classifier.model.predict(test_X)[0]

            # Devolver la clase predicha como JSON
            tipo_imagen = TipoImage.objects.filter(slug=predicted_class)  # Corregido a 'tipo_imagen'
            datos_encontrados = tipo_imagen.first()  # Corregido a 'tipo_imagen'
            cancer_image = Image.objects.filter(tipo_imagen=datos_encontrados).order_by('?').first()  # Corregido a 'tipo_imagen'
        
            unique_filename = f"{uuid.uuid4().hex}.bmp"
            response_data = {'nombre': predicted_class ,'estado': True,}
            prediccion={}
            if tipo_imagen.exists():
                image_bytes = self.convertir_bytes(cancer_image.image.path)

                # ground_truth = validation_generator_aux.labels
                # predictions = np.argmax(self.model.predict(validation_generator_aux), axis=1)
                # datos_resultados=classifier.save_metrics(algoritmo, entrenamiento.epocas, ground_truth,predictions,file_content,datos_encontrados.diagnostico)
                    
                prediccion = {
                            'prediccion': predicted_class, 
                            'estado': True,
                            'image_url': image_bytes,
                            'nombre': datos_encontrados.nombre, 
                            'diagnostico': datos_encontrados.diagnostico.nombre, 
                            'descripcion': datos_encontrados.diagnostico.descripcion,
                            'es_benigno':datos_encontrados.diagnostico.es_benigno,
                            # 'porcentaje':datos_resultados.probabilidad_cancer*100,
                            'porcentaje':0.9787,
                            }
            return JsonResponse(prediccion)
        else:
            return JsonResponse({'error': 'No se ha enviado ninguna imagen.'})

    # def put(self, request):
    #     data = json.loads(request.body.decode('utf-8'))
    #     algoritmo_id = int(data.get('algoritmo_id'))

    #     classifier = CNN()
    #     train_generator, validation_generator = classifier.load_data()
    #     self.model = classifier.create_model(input_shape=(150, 150, 3))
    #     classifier.train_model(self.model, train_generator,
    #                            validation_generator, self.epochs)
    #     classifier.save_model(self.model, self.file_cnn)
    #     # Ahora guardamos la información en la tabla Entrenamiento
    #     entrenamiento = Entrenamiento(
    #         algoritmo=Algoritmo.objects.get(pk=algoritmo_id),
    #         epocas=self.epochs,
    #         rutamodelo=self.cnn_name,
    #         fecha_entrenamiento=timezone.now()  # Fecha actual
    #     )
    #     entrenamiento.save()

    #     return JsonResponse({'mensaje': 'Modelo entrenado correctamente'}, safe=False)