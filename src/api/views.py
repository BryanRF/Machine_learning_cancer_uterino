import io
from pathlib import Path
from typing import Any
from django.http import JsonResponse
from django.views import View
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from .machine_learning.CNN import CNN
from .models import TipoImagen as TipoImage
from .models import Image
from django.core.files.base import ContentFile
import base64
import os
from pathlib import Path

class TipoImagenView(View):
    
    def __init__(self):
        super().__init__()
        self.train_generator = None  # Inicializa train_generator como None en el constructor
        self.model = None  # Inicializa model como None en el constructor
        # current_dir = os.path.dirname(os.path.abspath(__file__))
        # self.model_filename = os.path.join(current_dir, "machine_learning", "entrenamiento", "cancer_classifier_model.h5")
        self.model_filename = 'C:/xampp/htdocs/proyecto_cancer_cuello_uterino/src/api/machine_learning/entrenamiento/cancer_classifier_model.h5'
    
    @method_decorator(csrf_exempt)
    def dispatch(self, request, *args: Any, **kwargs: Any):
        return super().dispatch(request, *args, **kwargs)
    
    def put(self, request):
        epochs = 10
        classifier = CNN()
        # Carga los datos y entrena el modelo
        train_generator, validation_generator = classifier.load_data()
        self.model = classifier.create_model(input_shape=(150, 150, 3), num_classes=3)
        classifier.train_model(self.model, train_generator, validation_generator, epochs)
        # Guarda el modelo entrenado
        classifier.save_model(self.model, self.model_filename)
        return JsonResponse({'mensaje':'Modelo entrenado correctamente'}, safe=False)
    
    def get(self, request):
        # Consulta todas las eTipoImagen en la base de datos
        all_TipoImagen = TipoImage.objects.all()
        # Crea una lista para almacenar los datos de las eTipoImagen
        TipoImagen_list = []
        # Itera sobre todas las eTipoImagen y agrega sus datos a la lista
        for TipoImagen in all_TipoImagen:
            TipoImagen_data = {
                'name': TipoImagen.nombre,
                # Agrega más campos según sea necesario
            }
            
            TipoImagen_list.append(TipoImagen_data)
        # Retorna la lista de eTipoImagen como una respuesta JSON
        return JsonResponse({'TipoImagen': TipoImagen_list}, safe=False)
        
    def convertir_bytes(self,image_path):
        try:
            with open(image_path, 'rb') as image_file:
                image_bytes = image_file.read()
                return base64.b64encode(image_bytes).decode('utf-8')
        except Exception as e:
            # Manejar errores si la imagen no se puede leer
            return None
        
    def post(self, request):
        epochs = 10
        # Verifica si ya existe un modelo entrenado
        if Path(self.model_filename).is_file():
            # Si existe, carga el modelo entrenado
            classifier = CNN()
            self.model = classifier.load_model(self.model_filename)
            # Inicializa train_generator si no lo está
            if self.train_generator is None:
                self.train_generator, _ = classifier.load_data()  # Usamos _ para descartar validation_generator
        else:
            # Si no existe, crea una nueva instancia de CNN
            classifier = CNN()
            # Carga los datos y entrena el modelo
            train_generator, validation_generator = classifier.load_data()
            self.model = classifier.create_model(input_shape=(150, 150, 3), num_classes=3)
            classifier.train_model(self.model, train_generator, validation_generator, epochs)
            # Guarda el modelo entrenado
            classifier.save_model(self.model, self.model_filename)
            # Asigna train_generator a la instancia para futuros usos
            self.train_generator = train_generator
        # Continúa con la lógica para procesar la imagen y hacer la predicción
        file_content = request.FILES['image'].read()
        image_io = io.BytesIO(file_content)
        predicted_class = classifier.predict_cancer(self.model, image_io)
        class_names = list(self.train_generator.class_indices.keys())  # Usamos self.train_generator
        predicted_TipoImagen_slug = class_names[predicted_class]
        tipo_imagen = TipoImage.objects.filter(slug=predicted_TipoImagen_slug)  # Corregido a 'tipo_imagen'
        especie = tipo_imagen.first()  # Corregido a 'tipo_imagen'
        cancer_image = Image.objects.filter(tipo_imagen=especie).order_by('?').first()  # Corregido a 'tipo_imagen'
        image_bytes = self.convertir_bytes(cancer_image.image.path)
        if tipo_imagen.exists():  # Corregido a 'tipo_imagen'
            # Ahora, puedes acceder a los datos de la especie
            tipo_imagen_data = {  # Cambiado a minúsculas por convención
                'estado': True,
                'name': tipo_imagen.nombre,  # Cambiado a minúsculas por convención
                'image_url': image_bytes
            }
            tipo_imagen_list = [tipo_imagen_data]  # Cambiado a minúsculas por convención
        else:    
            return JsonResponse(mensaje='no se encontro ninguna especie con esas caracteristicas', data=predicted_TipoImagen_slug, code=400)  # Corregido a minúsculas por convención
        return JsonResponse(tipo_imagen_list, safe=False)  # Cambiado a minúsculas por convención
