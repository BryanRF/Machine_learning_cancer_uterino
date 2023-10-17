from django.contrib import admin
from .models import TipoImagen, Image, Diagnostico, Resultado, MetricasDesempeno, Algoritmo,Entrenamiento

admin.site.register(Image)
admin.site.register(TipoImagen)
admin.site.register(Diagnostico)
admin.site.register(Resultado)
admin.site.register(MetricasDesempeno)
admin.site.register(Algoritmo)
admin.site.register(Entrenamiento)