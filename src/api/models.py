from django.db import models
from django.utils.text import slugify
import os
import uuid

def image_path(instance, filename):
    unique_filename = f"{uuid.uuid4().hex}{os.path.splitext(filename)[1]}"
    slug = slugify(instance.tipo_imagen.nombre)
    return os.path.join('media', slug, unique_filename)


class TipoImagen(models.Model):
    tipo_id = models.AutoField(primary_key=True)
    nombre = models.CharField(max_length=100)
    slug = models.SlugField(max_length=100, unique=True)
    def save(self, *args, **kwargs):
        self.slug = slugify(self.nombre)
        super(TipoImagen, self).save(*args, **kwargs)
class Image(models.Model):
    tipo_imagen = models.ForeignKey(TipoImagen, on_delete=models.CASCADE)
    image = models.ImageField(upload_to=image_path)

    def __str__(self):
        return f"Imagen de {self.tipo_imagen.nombre}"
class Diagnostico(models.Model):
    diagnostico_id = models.AutoField(primary_key=True)
    nombre = models.CharField(max_length=100)

    def __str__(self):
        return self.nombre
class Resultados(models.Model):
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
