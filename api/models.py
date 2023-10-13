from django.db import models
from django.utils.text import slugify
import os
import uuid


def image_path(instance, filename):
    unique_filename = f"{uuid.uuid4().hex}{os.path.splitext(filename)[1]}"
    slug = slugify(instance.tipo_imagen.nombre)
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'media')
    data_dir= 'C:/BrayanHTEC/Machine_learning_cancer_uterino/media'
    return os.path.join(data_dir, slug, unique_filename)

class TipoImagen(models.Model):
    tipo_id = models.AutoField(primary_key=True)
    nombre = models.CharField(max_length=100)
    slug = models.SlugField(max_length=100, unique=True, null=True, blank=True)

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

class Diagnostico(models.Model):
    diagnostico_id = models.AutoField(primary_key=True)
    nombre = models.CharField(max_length=100)

    def __str__(self):
        return self.nombre
class Resultado(models.Model):
    resultado_id = models.AutoField(primary_key=True)
    probabilidad_cancer = models.FloatField()
    diagnostico = models.ForeignKey(Diagnostico, on_delete=models.CASCADE)

    def __str__(self):
        return f"Diagnóstico: {self.diagnostico.nombre}"
class MetricasDesempeno(models.Model):
    metrica_id = models.AutoField(primary_key=True)
    modelo = models.CharField(max_length=100)
    precision = models.FloatField()
    sensibilidad = models.FloatField()
    especificidad = models.FloatField()
    exactitud = models.FloatField()
    algoritmo = models.ForeignKey('Algoritmo', on_delete=models.CASCADE)  # Añade el campo de clave foránea
    datos_entrenados = models.PositiveIntegerField()  # Nuevo campo para la cantidad de datos entrenados

    def __str__(self):
        return f"Metricas del modelo {self.modelo}"
class Algoritmo(models.Model):
    algoritmo_id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=100)
    descripcion = models.CharField(max_length=500)
    abrebiatura = models.CharField(max_length=30)

    def __str__(self):
        return self.name