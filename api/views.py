import io
import numpy as np
from pathlib import Path
from typing import Any
from django.http import JsonResponse
from django.views import View
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from .machine_learning.CNN import CNN
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


class TipoImagenView(View):

    def __init__(self):
        super().__init__()
        self.train_generator = None  # Inicializa train_generator como None en el constructor
        self.model = None  # Inicializa model como None en el constructor
        self.epochs = 1
        # current_dir = os.path.dirname(os.path.abspath(__file__))
        # self.model_filename = os.path.join(current_dir, "machine_learning", "entrenamiento", "cancer_classifier_model.h5")
        self.model_filename = 'C:/BrayanHTEC/Machine_learning_cancer_uterino/api/machine_learning/entrenamiento/cancer_classifier_model.h5'

    @method_decorator(csrf_exempt)
    def dispatch(self, request, *args: Any, **kwargs: Any):
        return super().dispatch(request, *args, **kwargs)

    def put(self, request):
        data = json.loads(request.body.decode('utf-8'))
        algoritmo = data.get('algoritmo')
        self.epochs = int(data.get('epochs'))
        classifier = CNN()

        train_generator, validation_generator = classifier.load_data()
        self.model = classifier.create_model(input_shape=(150, 150, 3))
        classifier.train_model(self.model, train_generator,
                               validation_generator, self.epochs)
        classifier.save_model(self.model, self.model_filename)

        # Ahora guardamos la información en la tabla Entrenamiento
        entrenamiento = Entrenamiento(
            algoritmo=Algoritmo.objects.get(abrebiatura=algoritmo),
            epocas=self.epochs,
            rutamodelo=self.model_filename,
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
        data = json.loads(request.body.decode('utf-8'))
        algoritmo_id = data.get('id_algoritmo')
        algoritmo = Algoritmo.objects.get(pk=algoritmo_id)
        entrenamiento = Entrenamiento.objects.filter(
            algoritmo=algoritmo).order_by('-fecha_entrenamiento').first()
        self.epochs = entrenamiento.epocas if entrenamiento else None
        if self.epochs is not None:
            self.epochs = int(self.epochs)
        else:
            # No se encontraron entrenamientos para el algoritmo
            return JsonResponse(
                mensaje='Por favor, entrene el algoritmo primero.',
                status=400
            )

        classifier = CNN()
        # Verifica si ya existe un modelo entrenado
        if Path(self.model_filename).is_file():
            # Si existe, carga el modelo entrenado
            self.model = classifier.load_model(self.model_filename)
            # Inicializa train_generator si no lo está
            if self.train_generator is None:
                # Usamos _ para descartar validation_generator
                self.train_generator, _ = classifier.load_data()
        else:
            # Si no existe, crea una nueva instancia de CNN
            # Carga los datos y entrena el modelo
            train_generator, validation_generator = classifier.load_data()
            self.model = classifier.create_model(input_shape=(150, 150, 3))
            classifier.train_model(
                self.model, train_generator, validation_generator, self.epochs)
            # Guarda el modelo entrenado
            classifier.save_model(self.model, self.model_filename)
            # Asigna train_generator a la instancia para futuros usos
            self.train_generator = train_generator
        # Continúa con la lógica para procesar la imagen y hacer la predicción
        file_content = request.FILES['image'].read()
        image_io = io.BytesIO(file_content)
        predicted_class = classifier.predict_cancer(self.model, image_io)
        # Usamos self.train_generator
        class_names = list(self.train_generator.class_indices.keys())
        predicted_TipoImagen_slug = class_names[predicted_class]
        tipo_imagen = TipoImage.objects.filter(
            slug=predicted_TipoImagen_slug)  # Corregido a 'tipo_imagen'
        datos_encontrados = tipo_imagen.first()  # Corregido a 'tipo_imagen'

        cancer_image = Image.objects.filter(tipo_imagen=datos_encontrados).order_by(
            '?').first()  # Corregido a 'tipo_imagen'
        image_bytes = self.convertir_bytes(cancer_image.image.path)

        if tipo_imagen.exists():

            # Aqui tengo que realizar los guardados de metricas
            imagen_referencia_bytes = request.FILES['image'].read()
            imagen_prediccion_bytes = image_bytes.read()
            imagen_referencia_io = io.BytesIO(imagen_referencia_bytes)
            imagen_prediccion_io = io.BytesIO(imagen_prediccion_bytes)
            ground_truth = validation_generator.labels
        # Obtenemos las predicciones (predictions)
            predictions = np.argmax(
                self.model.predict(validation_generator), axis=1)
            CNN.save_metrics(algoritmo, self.epochs, ground_truth,
                             predictions, imagen_referencia_io, imagen_prediccion_io)
            prediccion = {'prediccion': class_names[predicted_class], 
                          'estado': True,
                          'image_url': image_bytes,
                          'nombre': datos_encontrados.nombre, 
                          'diagnostico': datos_encontrados.diagnostico.nombre, 
                          'descripcion': datos_encontrados.diagnostico.descripcion, }
        else:
            # Corregido a minúsculas por convención
            return JsonResponse(mensaje='no se encontro ninguna especie con esas caracteristicas', data=predicted_TipoImagen_slug, code=400)
        # Cambiado a minúsculas por convención
        return JsonResponse(prediccion, safe=False)
