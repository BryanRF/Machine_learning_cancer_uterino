from django.db import models
from django.utils.text import slugify
import os
import uuid
from datetime import date

def image_path(instance, filename):
    unique_filename = f"{uuid.uuid4().hex}{os.path.splitext(filename)[1]}"
    slug = slugify(instance.tipo_imagen.nombre)
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'media')
    data_dir= 'C:/BrayanHTEC/Machine_learning_cancer_uterino/media'
    return os.path.join(data_dir, slug, unique_filename)
def image_path2(instance, filename):
    unique_filename = f"{uuid.uuid4().hex}{os.path.splitext(filename)[1]}"
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'analisis')
    data_dir= 'C:/BrayanHTEC/Machine_learning_cancer_uterino/analisis'
    return os.path.join(data_dir, unique_filename)
class Diagnostico(models.Model):
    diagnostico_id = models.AutoField(primary_key=True)
    nombre = models.CharField(max_length=100)
    descripcion = models.TextField()
    es_benigno = models.BooleanField(default=False) 
    def __str__(self):
        return self.nombre
class TipoImagen(models.Model):
    tipo_id = models.AutoField(primary_key=True)
    nombre = models.CharField(max_length=100)
    slug = models.SlugField(max_length=100, unique=True, null=True, blank=True)
    diagnostico = models.ForeignKey(Diagnostico, on_delete=models.CASCADE)
    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.nombre)
        super(TipoImagen, self).save(*args, **kwargs)
    def __str__(self):
        return self.nombre
class Image(models.Model):
    tipo_imagen = models.ForeignKey(TipoImagen, on_delete=models.CASCADE)
    image = models.ImageField(upload_to=image_path)
    def __str__(self):
        return f"{self.image} | Imagen de {self.tipo_imagen.nombre}"

class MetricasDesempeno(models.Model):
    metrica_id = models.AutoField(primary_key=True)
    modelo = models.CharField(max_length=100)
    precision = models.DecimalField(max_digits=5, decimal_places=3)
    sensibilidad = models.DecimalField(max_digits=5, decimal_places=3)
    especificidad = models.DecimalField(max_digits=5, decimal_places=3)
    exactitud = models.DecimalField(max_digits=5, decimal_places=3)
    epocas = models.PositiveIntegerField()  
    algoritmo = models.ForeignKey('Algoritmo', on_delete=models.CASCADE)  

    def __str__(self):
        return f"Metricas del modelo {self.modelo}"
class Resultado(models.Model):
    resultado_id = models.AutoField(primary_key=True)
    probabilidad_cancer = models.FloatField(default=0)
    fecha = models.DateField(default=date.today)  # Campo de fecha
    imagen_analizada = models.ImageField(upload_to=image_path2) # Campo de imagen
    diagnostico = models.ForeignKey(Diagnostico, on_delete=models.CASCADE)

    def __str__(self):
        return f"Diagnóstico: {self.diagnostico.nombre}"
class Algoritmo(models.Model):
    algoritmo_id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=100)
    descripcion = models.CharField(max_length=500)
    abrebiatura = models.CharField(max_length=30)

    def __str__(self):
        return self.name
class Entrenamiento(models.Model):
    entrenamiento_id = models.AutoField(primary_key=True)
    algoritmo = models.ForeignKey(Algoritmo, on_delete=models.CASCADE)
    epocas = models.PositiveIntegerField()
    rutamodelo = models.CharField(max_length=255)
    fecha_entrenamiento = models.DateTimeField(auto_now_add=True)
    def delete(self, *args, **kwargs):
        # Eliminar el archivo asociado
        ruta_completa = os.path.join("machine_learning", "entrenamiento", self.rutamodelo)
        if os.path.exists(ruta_completa):
            os.remove(ruta_completa)
        
        super().delete(*args, **kwargs)
    def __str__(self):
        return f"Entrenamiento del algoritmo {self.algoritmo.name} (Epocas: {self.epocas})"
