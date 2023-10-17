from django.urls import path
from django.urls import re_path
from .views import TipoImagenView
from django.conf import settings
from django.conf.urls.static import static
urlpatterns =[
    re_path(r'^analisis/(?P<filename>[^/]+)$', TipoImagenView.serve_analisis, name='serve_analisis'),
    path('tipo_imagen', TipoImagenView.as_view(), name='image_list')
]


if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
