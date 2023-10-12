from django.contrib import admin
from .models import TipoImagen, Image, Diagnostico, Resultados, MetricasDesempeno
class ImageInline(admin.TabularInline):
    model = Image
    extra = 3  # Define cuántos campos de imagen adicionales se mostrarán
class TipoImagenAdmin(admin.ModelAdmin):
    inlines = [ImageInline]  # Asocia el InlineForm con el modelo

# Registra los modelos en el panel de administración
admin.site.register(TipoImagen, TipoImagenAdmin)
admin.site.register(Diagnostico)
admin.site.register(Resultados)
admin.site.register(MetricasDesempeno)
