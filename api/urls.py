from django.urls import path
from .views import TipoImagenView
urlpatterns =[
    path('tipo_imagen', TipoImagenView.as_view(),name='image_list')
]